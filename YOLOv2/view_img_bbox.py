import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
from torch.utils.data import DataLoader
from data import VOCDataset
from model import YOLOv2ResNet, YOLOv2
import config

# Argument parsing
parser = argparse.ArgumentParser(description="Visualize YOLO predictions")
parser.add_argument('--no-gt', dest='show_gt', action='store_false',
                    help='Disable ground truth box overlay')
parser.set_defaults(show_gt=True)
args = parser.parse_args()
SHOW_GT = args.show_gt

# Device setup
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}, show_gt={SHOW_GT}")

# Load model
model = YOLOv2().to(device)
state = torch.load("checkpoints/YOLOv2/best_model.pth", map_location=device)
model.load_state_dict(state["model_state_dict"])
model.eval()

# Dataset
dataset = VOCDataset("train")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Visualization setup
fig, ax = plt.subplots()

# NMS parameters
CONF_THRESH = 0.5
IOU_THRESH = 0.4


def draw_image(idx):
    ax.clear()
    img, target = dataset[idx]
    img_batch = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_batch).cpu().squeeze(0)

    # Prepare image
    img_np = img.permute(1, 2, 0).numpy()
    ax.imshow(img_np)
    ax.axis('off')

    # Cell sizes
    H, W = img_np.shape[:2]
    cell_w = W / config.S
    cell_h = H / config.S

    # Optionally draw Ground Truth boxes (red)
    if SHOW_GT:
        gt = target.view(config.S, config.S, config.B, 5 + config.C)
        for i in range(config.S):
            for j in range(config.S):
                for b in range(config.B):
                    tb = gt[i, j, b]
                    if tb[4] > 0.5:
                        cls_idx = tb[5:].argmax().item()
                        x, y, w, h = tb[0].item(), tb[1].item(), tb[2].item(), tb[3].item()
                        cx = (j + x) * cell_w
                        cy = (i + y) * cell_h
                        bw = w * config.IMG_SIZE[0]
                        bh = h * config.IMG_SIZE[1]
                        x1 = cx - bw / 2
                        y1 = cy - bh / 2
                        rect = patches.Rectangle((x1, y1), bw, bh,
                                                 linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x1, y1 - 5,
                                config.VOC_CLASSES[cls_idx],
                                color='white', backgroundcolor='r', fontsize=8)

    # Collect detections
    boxes, scores, labels = [], [], []
    preds = out.view(config.S, config.S, config.B, 5 + config.C)

    for i in range(config.S):
        for j in range(config.S):
            for b in range(config.B):
                p = preds[i, j, b]
                p_conf = p[4].item()
                if p_conf < CONF_THRESH:
                    continue

                cls_probs = p[5:]
                cls_idx = torch.argmax(cls_probs).item()
                score = p_conf * cls_probs[cls_idx].item()

                x, y, w, h = p[0].item(), p[1].item(), p[2].item(), p[3].item()
                cx = (j + x) * cell_w
                cy = (i + y) * cell_h
                bw = w * config.IMG_SIZE[0]
                bh = h * config.IMG_SIZE[1]
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2

                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                labels.append(cls_idx)

    if not boxes:
        plt.title("No detections above threshold")
        fig.canvas.draw()
        return

    boxes_t = torch.tensor(boxes, dtype=torch.float32, device=device)
    scores_t = torch.tensor(scores, dtype=torch.float32, device=device)

    # Per-class NMS
    keep_indices = []
    for cls in set(labels):
        inds = [i for i, l in enumerate(labels) if l == cls]
        cls_boxes = boxes_t[inds]
        cls_scores = scores_t[inds]
        keep = nms(cls_boxes, cls_scores, IOU_THRESH)
        keep_indices.extend([inds[i] for i in keep])

    # Draw kept detections (green)
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes[idx]
        cls = labels[idx]
        score = scores[idx]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5,
                f"{config.VOC_CLASSES[cls]}:{score:.2f}",
                color='white', backgroundcolor='g', fontsize=8)
        print(config.VOC_CLASSES[cls])

    plt.title(f"{'Gnd Truth (Red), ' if SHOW_GT else ''}Predictions (Green)")
    fig.canvas.draw()


# Interactive callback
index = [0]

def on_key(event):
    if event.key == ' ':
        index[0] = (index[0] + 1) % len(dataset)
        draw_image(index[0])

fig.canvas.mpl_connect('key_press_event', on_key)
# Initial draw
draw_image(index[0])
plt.show()
