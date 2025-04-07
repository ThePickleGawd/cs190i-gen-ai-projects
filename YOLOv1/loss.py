import torch
import torch.nn as nn

import config

class YOLOLoss(nn.Module):
    def __init__(self):
        super.__init__()

    def calc_iou(bbox1, bbox2):
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


    def forward(self, preds, targets):

        """
        SxS cells (7x7)
        Each cell is 30 depth
        First 5 is bounding box 1, Second is bounding box 2
        We can directly compare since the label data gives us both
        - The IOU is how much they intersect vs how much they total take up
        - We correspond them correctly based on the highest IOU
        The next 20 is class prediction
        """

        # From paper
        lambda_coord, lambda_noobj = 5, 0.5

        loss = 0.0

        for i in range(config.S):
            for j in range(config.S):
                target = targets[..., :]
                pred = preds[..., :]

                # No loss if no object in cell
                has_object = torch.any(target[-config.VOC_CLASSES:] != 0)

                ## MSE Bounding box loss

                # Get "responsible" bbox (the "1"^obj_ij)
                b_idx, ious = 0, []
                for b in range(config.B):
                    # iou for box index b (note: 5 features, but we don't want last one, conf)
                    pred_bbox = pred[b*5:(b+1)*5-1]
                    target_bbox = target[b*5,(b+1)*5-1]

                    iou = self.calc_iou(pred_bbox, target_bbox)
                    ious.append(iou)

                    if iou > iou:
                       b_idx = b

                if has_object:
                    pos_loss = (pred[b_idx*5] - target[b_idx*5])**2 + (pred[b_idx*5+1] - target[b_idx*5+1])** 2
                    loss += pos_loss
                

                ## MSE Probability loss (though I think cross-entropy is better...)
                pred_classes = pred[-config.VOC_CLASSES:]
                target_classes = target[-config.VOC_CLASSES:]
                loss += torch.sum((pred_classes - target_classes)**2)

                ## MSE Confidence loss
                gnd_truth_obj = torch.argmax(target_classes)
                for b in range(config.B):
                    pred_conf = iou[b] * pred_classes[gnd_truth_obj]
                    target_conf = iou[b] * 1

                    if b == b_idx:
                        loss += (pred_conf - target_conf)**2
                    else:
                        loss += lambda_noobj * (pred_conf - target_conf)**2
        
        N = pred[0] # batch size
        return loss / N