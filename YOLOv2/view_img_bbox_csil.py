import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
from torch.utils.data import DataLoader
from data import VOCDataset
from model import YOLOv2
import config

# Argument parsing (keep --no-gt if you want)
parser = argparse.ArgumentParser(description="Save YOLO predictions")
parser.add_argument('--no-gt', dest='show_gt', action='store_false',
                    help='Disable ground truth box overlay')
parser.set_defaults(show_gt=True)
args = parser.parse_args()
SHOW_GT = args.show_gt

# Prepare output dir
out_dir = "images/YOLOv2"
os.makedirs(out_dir, exist_ok=True)

# Device & model
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
model = YOLOv2().to(device)
state = torch.load("checkpoints/YOLOv2/best_model.pth", map_location=device)
model.load_state_dict(state["model_state_dict"])
model.eval()

# Data
dataset = VOCDataset("val")

# Plot boilerplate
fig, ax = plt.subplots()
CONF_THRESH, IOU_THRESH = 0.5, 0.4

def draw_image(idx):
    ax.clear()
    img, target = dataset[idx]
    img_batch = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_batch).cpu().squeeze(0)
    img_np = img.permute(1,2,0).numpy()
    ax.imshow(img_np); ax.axis('off')

    H, W = img_np.shape[:2]
    cw, ch = W/config.S, H/config.S

    # GT boxes
    if SHOW_GT:
        gt = target.view(config.S,config.S,config.B,5+config.C)
        for i in range(config.S):
            for j in range(config.S):
                for b in range(config.B):
                    tb = gt[i,j,b]
                    if tb[4] > .5:
                        cls = tb[5:].argmax().item()
                        x,y,w,h = tb[:4].tolist()
                        cx, cy = (j+x)*cw, (i+y)*ch
                        bw, bh = np.exp(w)*cw, np.exp(h)*ch
                        rect = patches.Rectangle((cx-bw/2, cy-bh/2), bw, bh,
                                                 lw=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(cx-bw/2, cy-bh/2-5, config.VOC_CLASSES[cls],
                                color='white', backgroundcolor='r', fontsize=8)

    # Predictions
    preds = out.view(config.S,config.S,config.B,5+config.C)
    boxes, scores, labels = [], [], []
    for i in range(config.S):
        for j in range(config.S):
            for b in range(config.B):
                p = preds[i,j,b]
                conf = p[4].item()
                if conf < CONF_THRESH: continue
                cls = p[5:].argmax().item()
                score = conf * p[5+cls].item()
                x,y,w,h = p[:4].tolist()
                cx, cy = (j+x)*cw, (i+y)*ch
                bw, bh = np.exp(w)*cw, np.exp(h)*ch
                x1,y1,x2,y2 = cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2
                boxes.append([x1,y1,x2,y2]); scores.append(score); labels.append(cls)

    if boxes:
        bt = torch.tensor(boxes, device=device)
        st = torch.tensor(scores, device=device)
        keep = []
        for cls in set(labels):
            inds = [i for i,l in enumerate(labels) if l==cls]
            k = nms(bt[inds], st[inds], IOU_THRESH)
            keep += [inds[i] for i in k]
        for i in keep:
            x1,y1,x2,y2 = boxes[i]
            rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                     lw=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{config.VOC_CLASSES[labels[i]]}:{scores[i]:.2f}",
                    color='white', backgroundcolor='g', fontsize=8)

    ax.set_title(f"{'GT+ ' if SHOW_GT else ''}Predictions")
    fig.tight_layout()

# Save first 3
for idx in range(3):
    draw_image(idx)
    fig.savefig(f"{out_dir}/{idx}.png", bbox_inches='tight')
