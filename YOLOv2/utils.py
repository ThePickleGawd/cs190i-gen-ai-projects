import torch
from collections import Counter
import os
import matplotlib.pyplot as plt
import config

def xywh_to_xyxy(box):
    """
    box: (N, S, S, B, 4) where x, y are relative to cell,
         and w, h are log-scale relative to cell size
    returns: (N, S, S, B, 4) in absolute xyxy image coordinates
    """
    N, S, _, B, _ = box.shape
    x, y, w, h = box.unbind(-1)

    x_cell_size = config.IMG_SIZE[0] / S
    y_cell_size = config.IMG_SIZE[1] / S

    # Compute grid offsets
    grid_y, grid_x = torch.meshgrid(
        torch.arange(S, device=box.device),
        torch.arange(S, device=box.device),
        indexing='ij'
    )
    grid_x = grid_x.view(1, S, S, 1).expand(N, S, S, B)
    grid_y = grid_y.view(1, S, S, 1).expand(N, S, S, B)

    # Convert to absolute positions
    x_abs = (grid_x + x) * x_cell_size
    y_abs = (grid_y + y) * y_cell_size
    w_abs = w * config.IMG_SIZE[0]
    h_abs = h * config.IMG_SIZE[1]

    x1 = x_abs - w_abs / 2
    y1 = y_abs - h_abs / 2
    x2 = x_abs + w_abs / 2
    y2 = y_abs + h_abs / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)

def batch_iou(a: torch.Tensor, b: torch.Tensor):
    """
    a: (N, S, S, B, 5+C)
    b: (N, S, S, B, 5+C)

    output: (N, S, S, B, B)

    Compares iou across every pred box and target box.
    We want to pick the index with best iou match   
    """

    a, b = a[..., :4], b[..., :4]

    # Get absolute image corners
    a = xywh_to_xyxy(a) # (N, S, S, B, 4)
    b = xywh_to_xyxy(b) # (N, S, S, B, 4)

    # Get area of a and b boxes
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    # Broadcast so we get B "iou options" for each box
    a = a.unsqueeze(4) # (N, S, S, B, 1, 4)
    b = b.unsqueeze(3) # (N, S, S, 1, B, 4)
    area_a = area_a.unsqueeze(4) # (N, S, S, B, 1, 4)
    area_b = area_b.unsqueeze(3) # (N, S, S, B, 1, 4)

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
    """
    preds: (N, S, S, B * (5+C))
    targets: (N, S, S, B * (5+C))

    Converts model output and targets to coco mAP format
    """

    N = preds.shape[0]

    preds = preds.view(N, config.S, config.S, config.B, 5 + config.C)
    targets = targets.view(N, config.S, config.S, config.B, 5 + config.C)

    # Get absolute image coordinates: (N, S, S, B, 5+C) => (N, S*S, B, 4)
    preds_xyxy = xywh_to_xyxy(preds[..., :4]).view(N, config.S * config.S, config.B, 4)
    targets_xyxy = xywh_to_xyxy(targets[..., :4]).view(N, config.S * config.S, config.B, 4)

    preds = preds.view(N, config.S * config.S, config.B, 5 + config.C)
    targets = targets.view(N, config.S * config.S, config.B, 5 + config.C)

    preds_list, targets_list = [], []

    # Save boxes, scores, labels for each img in batch
    for idx in range(N):
        pred_boxes, pred_labels, pred_scores = [], [], []
        target_boxes, target_labels = [], []

        for s in range(config.S * config.S):
            for b in range(config.B):
                pred_boxes.append(preds_xyxy[idx, s, b, :])
                pred_labels.append(torch.argmax(preds[idx, s, b, 5:]).item())
                pred_scores.append(preds[idx, s, b, 4])

                target_boxes.append(targets_xyxy[idx, s, b, :])
                target_labels.append(torch.argmax(targets[idx, s, b, 5:]).item())

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


def plot_training_metrics(train_losses, map_scores, train_times, model_name, save_dir="images"):
    os.makedirs(f"{save_dir}/{model_name}", exist_ok=True)

    fig, ax1 = plt.subplots()

    # Loss and mAP on left axis
    ax1.plot(train_losses, label='Loss', color='tab:blue')
    if map_scores:
        map_epochs = list(range(
            config.EVAL_INTERVAL,
            config.EVAL_INTERVAL + config.EVAL_INTERVAL * len(map_scores),
            config.EVAL_INTERVAL
        ))
        ax1.plot(map_epochs, [s * 100 / max(train_losses) for s in map_scores], label='mAP (%)', color='tab:green')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss / mAP", color='tab:blue')
    ax1.set_ylim(0, max(train_losses) * 1.3)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Training time on right axis
    ax2 = ax1.twinx()
    ax2.plot(train_times, label='Train Time (s)', color='tab:red')
    ax2.set_ylabel("Time (s)", color='tab:red')
    ax2.set_ylim(0, max(train_times) * 1.1)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title("Training Loss, mAP, and Time per Epoch")
    plt.savefig(f"{save_dir}/{model_name}/metrics.png")
    plt.close()