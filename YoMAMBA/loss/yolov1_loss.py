import torch
import torch.nn as nn
from utils.yolov1_utils import intersection_over_union as IOU

class YoloV1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_no_obj = 0.5
        self.lambda_obj = 5

    def forward(self, preds, target):
        mse_loss = nn.MSELoss(reduction="sum")

        preds = preds.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Find best box
        iou_bbox1 = IOU(preds[..., 20:24], target[..., 20:24])  # box1: x,y,w,h
        iou_bbox2 = IOU(preds[..., 25:29], target[..., 20:24])  # box2: x,y,w,h
        ious = torch.cat([iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)], dim=0)
        _, bestbox = torch.max(ious, dim=0)

        identity_obj_i = target[..., 24].unsqueeze(3)

        # BBox predictions: pick the best box using IOU
        boxpreds = identity_obj_i * (
            bestbox * preds[..., 25:29] + (1 - bestbox) * preds[..., 20:24]
        )
        boxtargets = identity_obj_i * target[..., 20:24]

        boxpreds[..., 2:4] = torch.sign(boxpreds[..., 2:4]) * torch.sqrt(torch.abs(boxpreds[..., 2:4] + 1e-6))
        boxtargets[..., 2:4] = torch.sqrt(boxtargets[..., 2:4])

        boxloss = mse_loss(
            torch.flatten(boxpreds, end_dim=-2),
            torch.flatten(boxtargets, end_dim=-2)
        )

        # Objectness confidence
        predbox_conf = bestbox * preds[..., 24:25] + (1 - bestbox) * preds[..., 29:30]
        objloss = mse_loss(
            torch.flatten(identity_obj_i * predbox_conf),
            torch.flatten(identity_obj_i * target[..., 24:25])
        )

        # Confidence Loss
        no_objloss = mse_loss(
            torch.flatten((1 - identity_obj_i) * preds[..., 24:25], start_dim=1),
            torch.flatten((1 - identity_obj_i) * target[..., 24:25], start_dim=1)
        )
        no_objloss += mse_loss(
            torch.flatten((1 - identity_obj_i) * preds[..., 29:30], start_dim=1),
            torch.flatten((1 - identity_obj_i) * target[..., 24:25], start_dim=1) # One target per box
        )

        # Classification loss
        classloss = mse_loss(
            torch.flatten(identity_obj_i * preds[..., :20], end_dim=-2),
            torch.flatten(identity_obj_i * target[..., :20], end_dim=-2)
        )

        loss = (
            self.lambda_obj * boxloss +
            objloss +
            self.lambda_no_obj * no_objloss +
            classloss
        )

        return loss
