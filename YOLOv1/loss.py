import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batch_iou
import config

class YOLOLoss(nn.Module):
    def __init__(self, lambda_class):
        self.lambda_class = lambda_class # Multiplier for classification

        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds, targets: (N, S, S, B*(5+C))
        """

        assert preds.shape == targets.shape

        N, S, *_ = preds.shape
        B, C = config.B, config.C

        preds = preds.view(N, S, S, B, 5 + C)
        targets = targets.view(N, S, S, B, 5 + C)

        loss = torch.tensor(0.0, device=preds.device)

        # Each box has B iou targets
        # Each box is responsible for one with best iou
        # gnd_truth is targets but the box is at correct spot for the pred
        ious = batch_iou(preds, targets) # (N, S, S, B, B)
        responsible = torch.argmax(ious, dim=-1, keepdim=True) # (N, S, S, B, 1)
        responsible = responsible.expand(-1, -1, -1, -1, targets.size(-1)) # (N, S, S, B, 5+C)
        gnd_truth = torch.gather(targets, dim=3, index=responsible) # (N, S, S, B, 5+C)

        # From paper
        lambda_coord, lambda_noobj = 5, 0.5
        obj_ij = (gnd_truth[..., 4] > 0) # (N, S, S, B)

        ## Bounding Box Loss
        
        # x,y loss
        loss += lambda_coord *  torch.sum(obj_ij.float() * (gnd_truth[..., 0] - preds[..., 0]) ** 2)
        loss += lambda_coord * torch.sum(obj_ij.float() * (gnd_truth[..., 1] - preds[..., 1]) ** 2)

        # w,h loss
        loss += lambda_coord * torch.sum(obj_ij.float() * (torch.sqrt(gnd_truth[..., 2]) - torch.sqrt(preds[..., 2])) ** 2)
        loss += lambda_coord * torch.sum(obj_ij.float() * (torch.sqrt(gnd_truth[..., 3]) - torch.sqrt(preds[..., 3])) ** 2)

        ## Confidence Loss

        conf_preds = preds[..., 4]
        conf_targets = gnd_truth[..., 4]
        loss += torch.sum(obj_ij.float() * (conf_targets - conf_preds) ** 2)
        loss += lambda_noobj * torch.sum((1 - obj_ij.float()) * (conf_targets - conf_preds) ** 2)

        ## Classification Loss

        # Use cross entropy instead of squared error for classification
        class_preds = preds[..., 5:][obj_ij]
        class_targets = gnd_truth[..., 5:][obj_ij].argmax(-1)
        loss += self.lambda_class * F.cross_entropy(class_preds.reshape(-1, C), class_targets.reshape(-1))

        return loss / N
    

class YOLOV2Loss(nn.Module):
    def __init__(self, lambda_class):
        super().__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_class = lambda_class

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds, targets: (N, S, S, B*(5+C))
        """
        N, S, _, _ = preds.shape
        B, C = config.B, config.C

        preds = preds.view(N, S, S, B, 5 + C)
        targets = targets.view(N, S, S, B, 5 + C)

        # Get IoU for each pred-target pair
        ious = batch_iou(preds, targets)  # (N, S, S, B, B)
        best_target_idx = torch.argmax(ious, dim=-1)  # (N, S, S, B)

        # For each pred box, find corresponding best-matching target box
        best_target_idx = best_target_idx.unsqueeze(-1)  # (N, S, S, B, 1)
        matched_targets = torch.gather(
            targets, 3, best_target_idx.expand(-1, -1, -1, -1, 5 + C)
        )

        # Object mask
        obj_mask = matched_targets[..., 4] > 0  # shape: (N, S, S, B)

        # Localization Loss (x, y, w, h)
        xy_loss = self.lambda_coord * F.mse_loss(
            preds[obj_mask][..., 0:2],
            matched_targets[obj_mask][..., 0:2],
            reduction="sum"
        )
        wh_loss = self.lambda_coord * F.mse_loss(
            torch.sqrt(torch.abs(preds[obj_mask][..., 2:4] + 1e-6)),
            torch.sqrt(torch.abs(matched_targets[obj_mask][..., 2:4] + 1e-6)),
            reduction="sum"
        )

        # Confidence Loss
        conf_loss_obj = F.mse_loss(
            preds[obj_mask][..., 4],
            matched_targets[obj_mask][..., 4],
            reduction="sum"
        )
        conf_loss_noobj = self.lambda_noobj * F.mse_loss(
            preds[~obj_mask][..., 4],
            matched_targets[~obj_mask][..., 4],
            reduction="sum"
        )

        # Classification Loss (only use one of the B boxes per cell)
        class_loss = self.lambda_class * F.cross_entropy(
            preds[obj_mask][..., 5:],  # logits
            torch.argmax(targets[obj_mask][..., 5:], dim=-1),  # class indices
            reduction="sum"
        )

        total_loss = xy_loss + wh_loss + conf_loss_obj + conf_loss_noobj + class_loss
        return total_loss
