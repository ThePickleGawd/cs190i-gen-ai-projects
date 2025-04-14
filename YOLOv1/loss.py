import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batch_iou
import config

class YOLOLoss(nn.Module):
    def __init__(self):
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

        # Compute IoUs between all predicted boxes and all GT boxes in each cell
        ious = batch_iou(preds, targets)  # (N, S, S, B_pred, B_gt)

        # Get best predicted box for each target box
        best_pred = torch.argmax(ious, dim=3)  # (N, S, S, B_gt)

        # Build a responsible mask: (N, S, S, B)
        responsible_mask = torch.zeros_like(preds[..., 0])  # (N, S, S, B)
        for b in range(B):
            responsible_mask.scatter_(3, best_pred[..., b:b+1], 1)

        obj_ij = responsible_mask.bool()  # (N, S, S, B)

        # Extract responsible predictions and targets
        preds_resp = preds[obj_ij]  # (num_responsible, 5+C)
        targets_resp = targets[obj_ij]  # (num_responsible, 5+C)

        # Coefficients
        lambda_coord = 5
        lambda_noobj = 0.5

        # Localization loss
        loss += lambda_coord * F.mse_loss(preds_resp[:, 0], targets_resp[:, 0], reduction='sum')
        loss += lambda_coord * F.mse_loss(preds_resp[:, 1], targets_resp[:, 1], reduction='sum')

        # w, h with sqrt
        loss += lambda_coord * F.mse_loss(
            torch.sqrt(torch.exp(preds_resp[:, 2])),
            torch.sqrt(torch.exp(targets_resp[:, 2])),
            reduction='sum'
        )
        loss += lambda_coord * F.mse_loss(
            torch.sqrt(torch.exp(preds_resp[:, 3])),
            torch.sqrt(torch.exp(targets_resp[:, 3])),
            reduction='sum'
        )

        # Confidence loss
        loss += F.mse_loss(preds_resp[:, 4], targets_resp[:, 4], reduction='sum')

        # No-object loss (where obj_ij is false)
        noobj_preds = preds[~obj_ij][..., 4]
        noobj_targets = targets[~obj_ij][..., 4]
        loss += lambda_noobj * F.mse_loss(noobj_preds, noobj_targets, reduction='sum')

        # Classification loss
        if preds_resp.shape[0] > 0:
            class_preds = preds_resp[:, 5:]
            class_targets = targets_resp[:, 5:].argmax(dim=1)
            loss += F.cross_entropy(class_preds, class_targets, reduction='sum')

        return loss / N