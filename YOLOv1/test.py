import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

from data import VOCDataset
from model import YOLO
import config


model = YOLO().to(config.device)
checkpoint = torch.load(f"checkpoints/checkpoint_epoch_9.pth")
model.load_state_dict(checkpoint['model_state_dict'])

dataset = VOCDataset("val")

dataloader = DataLoader(dataset, batch_size=1)

img, target = next(iter(dataloader))
img, target = img.to(config.device), target.to(config.device)

out = model(img)

S = out.shape[1]
B = 2  # adjust as needed
C = (out.shape[-1] // B) - 5

img = img.squeeze().permute(1, 2, 0).cpu().numpy()  # CHW to HWC
img_h, img_w = img.shape[:2]

fig, ax = plt.subplots()
ax.imshow(img)

# Loop through grid
for i in range(S):
    for j in range(S):
        for b in range(B):
            start = b * (5 + C)
            x, y, w, h, conf = target[0, i, j, start:start+5]

            if conf < 0.5:
                continue  # skip low-confidence boxes

            classes = target.view(-1, config.S, config.S, config.B, 5+config.C)
            print(classes[0, i, j, :, 5:])

            # x, y are relative to cell → convert to global coordinates
            box_x = (j + x.item()) / S * img_w
            box_y = (i + y.item()) / S * img_h
            box_w = w.item() * img_w
            box_h = h.item() * img_h

            # Convert center x,y to top-left corner
            top_left_x = box_x - box_w / 2
            top_left_y = box_y - box_h / 2

            rect = patches.Rectangle((top_left_x, top_left_y), box_w, box_h,
                                     linewidth=2, edgecolor='r', facecolor='none')
            print(box_x, box_y, box_w, box_h)
            ax.add_patch(rect)

plt.show()