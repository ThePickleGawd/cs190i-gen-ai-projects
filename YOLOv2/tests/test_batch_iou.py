import torch
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import IMG_SIZE  # Assumes IMG_SIZE is defined in config.py
from utils import batch_iou, xywh_to_xyxy

# Setup
N, S, B, C = 1, 1, 2, 0  # 1 image, 1x1 grid, 2 boxes
cell_size = IMG_SIZE[0] / S

# Boxes: (center x, y, log(w), log(h)) + dummy confidence + classes
box_a = torch.tensor([[[[[0.5, 0.5, math.log(0.5), math.log(0.5)],
                         [0.5, 0.5, math.log(1.0), math.log(1.0)]]]]], dtype=torch.float32)
box_b = torch.tensor([[[[[0.5, 0.5, math.log(0.5), math.log(0.5)],
                         [0.6, 0.5, math.log(0.4), math.log(0.4)]]]]], dtype=torch.float32)

# Pad with dummy conf + class
box_a = torch.cat([box_a, torch.zeros(N, S, S, B, 1 + C)], dim=-1)
box_b = torch.cat([box_b, torch.zeros(N, S, S, B, 1 + C)], dim=-1)

# Compute IOU
ious = batch_iou(box_a, box_b)
print("IOU matrix:\n", ious.squeeze().round(decimals=3))

# Visualization
def plot_boxes(ax, boxes, color, linestyle):
    abs_boxes = xywh_to_xyxy(boxes[..., :4])
    for i in range(B):
        x1, y1, x2, y2 = abs_boxes[0, 0, 0, i]
        rect = patches.Rectangle(
            (x1.item(), y1.item()), (x2 - x1).item(), (y2 - y1).item(),
            linewidth=2, edgecolor=color, linestyle=linestyle, facecolor='none'
        )
        ax.add_patch(rect)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, IMG_SIZE[0])
ax.set_ylim(0, IMG_SIZE[1])
ax.set_title("Pred (Red) vs Target (Blue)")
ax.set_aspect('equal')
plot_boxes(ax, box_a, color='red', linestyle='-')
plot_boxes(ax, box_b, color='blue', linestyle='--')
plt.gca().invert_yaxis()
plt.show()
