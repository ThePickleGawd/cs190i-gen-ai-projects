import torch
from collections import Counter
import config

def xywh_to_xyxy(box):
    x, y, w, h = box.unbind(-1)
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def batch_iou(a: torch.Tensor, b: torch.Tensor):
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

def batch_to_mAP_list(preds: torch.Tensor, targets: torch.Tensor):
    N = preds.shape[0]
    preds = preds.view(N, config.S * config.S, config.B, 5 + config.C)
    targets = targets.view(N, config.S * config.S, config.B, 5 + config.C)

    preds_list, targets_list = [], []

    for idx in range(N):
        pred_boxes, pred_labels, pred_scores = [], [], []
        target_boxes, target_labels = [], []

        for s in range(config.S * config.S):
            for b in range(config.B):
                pred_bbox = preds[idx, s, b]
                target_bbox = targets[idx, s, b]

                pred_boxes.append(xywh_to_xyxy(pred_bbox[:4]))
                pred_labels.append(torch.argmax(pred_bbox[5:]).item())
                pred_scores.append(pred_bbox[4])

                target_boxes.append(xywh_to_xyxy(target_bbox[:4]))
                target_labels.append(torch.argmax(target_bbox[5:]).item())

        preds_list.append({
            "boxes": torch.stack(pred_boxes),
            "scores": torch.stack(pred_scores),
            "labels": torch.tensor(pred_labels)
        })

        targets_list.append({
            "boxes": torch.stack(target_boxes),
            "labels": torch.tensor(target_labels)
        })

    return preds_list, targets_list