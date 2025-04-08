import torch
import torch.nn as nn

import config

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def calc_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        tl1 = (x1 - w1 / 2, y1 - h1 / 2)  # top-left
        br1 = (x1 + w1 / 2, y1 + h1 / 2)  # bottom-right

        tl2 = (x2 - w2 / 2, y2 - h2 / 2)
        br2 = (x2 + w2 / 2, y2 + h2 / 2)

        # Intersection, Union, IOU
        intersect_w = max(0, min(br1[0], br2[0]) - max(tl1[0], tl2[0]))
        intersect_h = max(0, min(br1[1], br2[1]) - max(tl1[1], tl2[1]))
        intersect = intersect_w * intersect_h

        union = w1*h1 + w2*h2 - intersect

        iou = intersect/union

        return iou
    
    def batch_iou(self, a, b):
        a, b = a[..., :4], b[..., :4]

        # TODO: Get area by w*h first, then conver to xyxy

        def xywh_to_xyxy(box):
            x, y, w, h = box.unbind(-1)
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            return torch.stack([x1, y1, x2, y2], dim=-1)
        
        # We want to broadcast so we get B "iou options" for each box
        a = xywh_to_xyxy(a).unsqueeze(4)  # (N, S, S, B, 1, 4)
        b = xywh_to_xyxy(b).unsqueeze(3)  # (N, S, S, 1, B, 4)

        # Overlapping intesection points
        inter_x1 = torch.max(a[..., 0], b[..., 0])
        inter_y1 = torch.max(a[..., 1], b[..., 1])
        inter_x2 = torch.min(a[..., 2], b[..., 2])
        inter_y2 = torch.min(a[..., 3], b[..., 3])

        # IOU
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter = inter_w * inter_h

        union = inter_w*inter_h - inter

        return inter/union



    def forward(self, preds, targets):
        """
        (N, S, S, B*(5+C))

        SxS cells (7x7)
        Each cell is B*(5+C) depth
        First 5 is bounding box, Next C are class. Repeate B times
        We can directly compare since the label data gives us both
        - The IOU is how much they intersect vs how much they total take up
        - We correspond them correctly based on the highest IOU
        The next 20 is class prediction
        """

        assert preds.shape == target.shape

        N, S, *_ = preds.shape
        B, C = config.B, config.C

        preds = preds.view(N, S, S, B, 5 + C)
        targets = targets.view(N, S, S, B, 5 + C)

        loss = 0.0

        # From paper
        lambda_coord, lambda_noobj = 5, 0.5
        obj_i = torch.zeros((N, S, S), dtype=torch.bool) # Whether grid cell has an object
        obj_ij = torch.zeros((N, S, S, B), dtype=torch.bool) # Whether this bounding box is responsible
        

        for i in range(S):
            for j in range(S):
                target = targets[:, i, j, :]
                pred = preds[:, i, j, :]

                # No loss if no object in cell

                ## MSE Bounding box loss

                # Get "responsible" bbox (the "1"^obj_ij)
                b_idx, ious = 0, []
                for b in range(B):
                    # iou for box index b (note: 5 features, but we don't want last one, conf)
                    pred_bbox = pred[b*5:(b+1)*5-1]
                    target_bbox = target[b*5,(b+1)*5-1]

                    print(pred_bbox.shape)

                    iou = self.calc_iou(pred_bbox, target_bbox)
                    ious.append(iou)

                    if iou > iou:
                       b_idx = b

                if has_object:
                    pos_loss = (pred[b_idx*5] - target[b_idx*5])**2 + (pred[b_idx*5+1] - target[b_idx*5+1])** 2
                    loss += pos_loss
                
                ## MSE Probability loss (though I think cross-entropy is better...)
                pred_classes = pred[-C:]
                target_classes = target[-C:]
                loss += torch.sum((pred_classes - target_classes)**2)

                ## MSE Confidence loss
                gnd_truth_obj = torch.argmax(target_classes)
                for b in range(B):
                    pred_conf = iou[b] * pred_classes[gnd_truth_obj]
                    target_conf = iou[b] * 1

                    if b == b_idx:
                        loss += (pred_conf - target_conf)**2
                    else:
                        loss += lambda_noobj * (pred_conf - target_conf)**2
        
        N = pred[0] # batch size
        return loss / N 