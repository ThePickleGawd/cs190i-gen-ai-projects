import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from data import VOCDataset
from model import YOLOv1ResNet
import config

# Load model
model = YOLOv1ResNet().to(config.device)
state_dict = torch.load("checkpoints/YOLOv1ResNet/best_model.pth", map_location=config.device)
model.load_state_dict(state_dict["model_state_dict"])
model.eval()

# Load dataset
dataset = VOCDataset("val")

fig, ax = plt.subplots()

def draw_image(idx):
    ax.clear()

    img, target = dataset[idx]
    img = img.unsqueeze(0)
    target = target.unsqueeze(0)

    with torch.no_grad():
        out = model(img.to(config.device)).cpu()

    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    img_h, img_w = img_np.shape[:2]
    x_cell_size = img_w / config.S
    y_cell_size = img_h / config.S

    target = target[0].view(config.S, config.S, config.B, 5 + config.C)
    pred = out[0].view(config.S, config.S, config.B, 5 + config.C)

    ax.imshow(img_np)
    for i in range(1, config.S):
        ax.axhline(i * y_cell_size, color='white', linestyle='--', linewidth=0.5)
        ax.axvline(i * x_cell_size, color='white', linestyle='--', linewidth=0.5)

    def draw_box(i, j, box, color, label=None):
        x, y, w, h = box[:4]
        box_x = (j + x) * x_cell_size
        box_y = (i + y) * y_cell_size
        box_w = np.exp(w) * x_cell_size
        box_h = np.exp(h) * y_cell_size
        top_left_x = box_x - box_w / 2
        top_left_y = box_y - box_h / 2

        rect = patches.Rectangle((top_left_x, top_left_y), box_w, box_h,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.plot(box_x, box_y, marker='o', color=color, markersize=3)
        if label:
            ax.text(top_left_x, top_left_y - 5, label,
                    color='white', fontsize=8, backgroundcolor=color)

    for i in range(config.S):
        for j in range(config.S):
            for b in range(config.B):
                t_box = target[i, j, b]
                p_box = pred[i, j, b]

                if t_box[4] > 0.5:
                    t_class_idx = torch.argmax(t_box[5:]).item()
                    t_label = config.VOC_CLASSES[t_class_idx]
                    draw_box(i, j, t_box, 'r', label=t_label)

                if p_box[4] > 0.5:
                    p_class_idx = torch.argmax(p_box[5:]).item()
                    p_label = config.VOC_CLASSES[p_class_idx]
                    draw_box(i, j, p_box, 'g', label=p_label)

    ax.set_title("Red = Ground Truth, Green = Prediction + Labels")
    ax.axis('off')
    fig.canvas.draw()

index = [0]  # Mutable container to allow update inside callback

def on_key(event):
    if event.key == ' ':
        index[0] = torch.randint(0, len(dataset), (1,)).item()
        draw_image(index[0])

fig.canvas.mpl_connect('key_press_event', on_key)
draw_image(index[0])
plt.show()
