import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

from data import VOCDataset
from model import YOLO
import config


# model = YOLO().to(config.device)
# checkpoint = torch.load(f"checkpoints/checkpoint_epoch_9.pth")
# model.load_state_dict(checkpoint['model_state_dict'])

dataset = VOCDataset("val")

# dataloader = DataLoader(dataset, batch_size=1)
import random

idx = random.randint(0, len(dataset) - 1)
img, target = dataset[idx]
img = img.unsqueeze(0)  # Add batch dimension
target = target.unsqueeze(0)

# out = model(img)

# S = out.shape[1]
# B = 2  # adjust as needed
# C = (out.shape[-1] // B) - 5

img = img.squeeze().permute(1, 2, 0).cpu().numpy()  # CHW to HWC
img_h, img_w = img.shape[:2]
x_cell_size = img_w / config.S 
y_cell_size = img_h / config.S

fig, ax = plt.subplots()
ax.imshow(img)

# Draw grid lines
for i in range(1, config.S):
    ax.axhline(i * y_cell_size, color='white', linestyle='--', linewidth=0.5)
    ax.axvline(i * x_cell_size, color='white', linestyle='--', linewidth=0.5)

# Loop through grid
for i in range(config.S):
    for j in range(config.S):
        for b in range(config.B):
            start = b * (5 + config.C)
            x, y, w, h, conf = target[0, i, j, start:start+5]

            if conf < 0.5:
                continue  # skip low-confidence boxes

            classes = target.view(-1, config.S, config.S, config.B, 5+config.C)
            print(classes[0, i, j, :, :])

            # x, y are relative to cell → convert to global coordinates
            box_x = ((j + x.item()) * x_cell_size) 
            box_y = ((i + y.item()) * y_cell_size)
            box_w = np.exp(w.item()) * x_cell_size
            box_h = np.exp(h.item()) * y_cell_size

            # Convert center x,y to top-left corner
            top_left_x = box_x - box_w / 2
            top_left_y = box_y - box_h / 2

            rect = patches.Rectangle((top_left_x, top_left_y), box_w, box_h,
                                     linewidth=2, edgecolor='r', facecolor='none')
            
            rect = patches.Rectangle((top_left_x, top_left_y), box_w, box_h,
                         linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Draw center point
            ax.plot(box_x, box_y, 'ro', markersize=3)


plt.show()