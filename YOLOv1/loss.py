import torch
import torch.nn as nn
import torch.nn.functional as F

import config

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def batch_iou(self, a, b):
        """
        a, b: (N, S, S, B, 5+C)
        output: (N, S, S, B, B)

        Compares iou across every pred box and target box.
        We want to pick the index with best iou match   
        """

        a, b = a[..., :4], b[..., :4]

        # Get area of a and b boxes
        area_a = (a[..., 2] * a[..., 3]).unsqueeze(4)  # shape: (N, S, S, B, 1)
        area_b = (b[..., 2] * b[..., 3]).unsqueeze(3)  # shape: (N, S, S, 1, B)


        def xywh_to_xyxy(box):
            x, y, w, h = box.unbind(-1)
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            return torch.stack([x1, y1, x2, y2], dim=-1)
        
        # Get corners. Also, broadcast so we get B "iou options" for each box
        a = xywh_to_xyxy(a).unsqueeze(4)  # (N, S, S, B, 1, 4)
        b = xywh_to_xyxy(b).unsqueeze(3)  # (N, S, S, 1, B, 4)

        # Overlapping intesection points
        inter_x1 = torch.max(a[..., 0], b[..., 0])
        inter_y1 = torch.max(a[..., 1], b[..., 1])
        inter_x2 = torch.min(a[..., 2], b[..., 2])
        inter_y2 = torch.min(a[..., 3], b[..., 3])

        # Intersection area
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter = inter_w * inter_h

        # IOU
        union = area_a + area_b - inter
        ious = inter / union
        ious = torch.where(union > 0, ious, torch.zeros_like(ious)) # No NaN

        return ious



    # def forward(self, preds: torch.Tensor, targets: torch.Tensor):
    #     """
    #     preds, targets: (N, S, S, B*(5+C))
    #     """

    #     assert preds.shape == targets.shape

    #     N, S, *_ = preds.shape
    #     B, C = config.B, config.C

    #     preds = preds.view(N, S, S, B, 5 + C)
    #     targets = targets.view(N, S, S, B, 5 + C)

    #     loss = torch.tensor(0.0, device=preds.device)

    #     # Each box has B iou targets
    #     # Each box is responsible for one with best iou
    #     # gnd_truth is targets but the box is at correct spot for the pred
    #     ious = self.batch_iou(preds, targets)
    #     responsible = torch.argmax(ious, dim=-1, keepdim=True)
    #     responsible = responsible.expand(-1, -1, -1, -1, targets.size(-1))
    #     gnd_truth = torch.gather(targets, dim=3, index=responsible) # (N, S, S, B, 5+C)

    #     # TODO: Maybe we actually can't have duplicates

    #     # From paper
    #     lambda_coord, lambda_noobj = 5, 0.5
    #     obj_ij = (gnd_truth[..., 4] > 0) # (N, S, S, B)

    #     ## Bounding Box Loss
        
    #     # x,y loss
    #     loss += lambda_coord * torch.sum(obj_ij.float() * (gnd_truth[..., 0] - preds[..., 0]) ** 2)
    #     loss += lambda_coord * torch.sum(obj_ij.float() * (gnd_truth[..., 1] - preds[..., 1]) ** 2)
        
    #     # w, h loss
    #     loss += lambda_coord * torch.sum(obj_ij.float() * ((gnd_truth[..., 2].clamp(min=config.EPSILON).sqrt() - preds[..., 2].clamp(min=config.EPSILON).sqrt()) ** 2))
    #     loss += lambda_coord * torch.sum(obj_ij.float() * ((gnd_truth[..., 3].clamp(min=config.EPSILON).sqrt() - preds[..., 3].clamp(min=config.EPSILON).sqrt()) ** 2))

    #     ## Confidence Loss

    #     conf_preds = preds[..., 4]
    #     conf_targets = gnd_truth[..., 4]

    #     # Note difference. Only apply lambda_noobj when no obj in entire cell
    #     loss += torch.sum(obj_ij.float() * (conf_targets - conf_preds) ** 2)
    #     loss += lambda_noobj * torch.sum((1 - obj_ij.float()) * (conf_targets - conf_preds) ** 2)

    #     ## Classification Loss

    #     # Use cross entropy instead of squared error for classification
    #     class_preds = preds[..., 5:][obj_ij]
    #     class_targets = gnd_truth[..., 5:][obj_ij].argmax(-1)
    #     loss += F.cross_entropy(class_preds.reshape(-1, C), class_targets.reshape(-1))

    #     return loss / N

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds, targets: (N, S, S, B*(5+C))
        """

        assert preds.shape == targets.shape

        N, S, *_ = preds.shape
        B, C = config.B, config.C

        preds = preds.view(N, S, S, B, 5 + C)
        targets = targets.view(N, S, S, B, 5 + C)

        print("Preds shape:", preds.shape)
        print("Targets shape:", targets.shape)

        loss = torch.tensor(0.0, device=preds.device)

        ious = self.batch_iou(preds, targets)
        print("IOUs shape:", ious.shape)

        responsible = torch.argmax(ious, dim=-1, keepdim=True)
        print("Responsible indices shape:", responsible.shape)

        responsible = responsible.expand(-1, -1, -1, -1, targets.size(-1))
        gnd_truth = torch.gather(targets, dim=3, index=responsible)
        print("Ground truth shape:", gnd_truth.shape)

        lambda_coord, lambda_noobj = 5, 0.5
        obj_ij = (gnd_truth[..., 4] > 0)
        print("Object mask shape:", obj_ij.shape)
        print("Number of object cells:", obj_ij.sum().item())

        # Bounding Box Loss
        loss += lambda_coord * torch.sum(obj_ij.float() * (gnd_truth[..., 0] - preds[..., 0]) ** 2)
        loss += lambda_coord * torch.sum(obj_ij.float() * (gnd_truth[..., 1] - preds[..., 1]) ** 2)

        loss += lambda_coord * torch.sum(obj_ij.float() * (
            (gnd_truth[..., 2].clamp(min=config.EPSILON).sqrt() - preds[..., 2].clamp(min=config.EPSILON).sqrt()) ** 2))
        loss += lambda_coord * torch.sum(obj_ij.float() * (
            (gnd_truth[..., 3].clamp(min=config.EPSILON).sqrt() - preds[..., 3].clamp(min=config.EPSILON).sqrt()) ** 2))

        # Confidence Loss
        conf_preds = preds[..., 4]
        conf_targets = gnd_truth[..., 4]

        print("Conf preds sample:", conf_preds[0, 0, 0])
        print("Conf targets sample:", conf_targets[0, 0, 0])

        loss += torch.sum(obj_ij.float() * (conf_targets - conf_preds) ** 2)
        loss += lambda_noobj * torch.sum((1 - obj_ij.float()) * (conf_targets - conf_preds) ** 2)

        # Classification Loss
        class_preds = preds[..., 5:][obj_ij]
        class_targets = gnd_truth[..., 5:][obj_ij].argmax(-1)

        print("Class preds shape:", class_preds.shape)
        print("Class targets shape:", class_targets.shape)

        loss += F.cross_entropy(class_preds.reshape(-1, C), class_targets.reshape(-1))

        print("Total loss:", loss.item())

        return loss / N


def test_batch_iou():
    loss_fn = YOLOLoss()

    # Shape: (N=1, S=1, S=1, B=2, 5+C)
    preds = torch.tensor([[[[
        [0.5, 0.5, 1.0, 1.0, 1.0],  # Perfect box
        [0.5, 0.5, 0.5, 0.5, 1.0]   # Smaller box
    ]]]])

    targets = torch.tensor([[[[
        [0.5, 0.5, 1.0, 1.0, 1.0],  # Same as first pred
        [0.6, 0.6, 0.4, 0.4, 1.0]   # Slightly offset
    ]]]])

    # Pad with dummy class data
    C = config.C
    preds = torch.cat([preds, torch.rand(1, 1, 1, 2, C)], dim=-1)
    targets = torch.cat([targets, torch.rand(1, 1, 1, 2, C)], dim=-1)

    loss = loss_fn(preds, targets)

if __name__ == "__main__":
    test_batch_iou()