import torch
import torchvision


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