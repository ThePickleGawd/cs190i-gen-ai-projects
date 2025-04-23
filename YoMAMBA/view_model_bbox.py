import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.yolov1_utils import non_max_suppression, cellboxes_to_boxes, draw_bounding_box
from models.yolov1_resnet18 import YoloV1_Resnet18
from models.yolov1_mamba import YoloV1_Mamba
from data import VOCDataset  # Make sure this uses split="val"

# Device and transform
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = T.Compose([T.ToTensor()])

# Select model
use_resnet18_backbone = True
use_mamba_backbone = False
count = 30

if use_mamba_backbone:
    current_model = "mamba"
    model = YoloV1_Mamba(S=7, B=2, C=20).to(device)
elif use_resnet18_backbone:
    current_model = "resnet18"
    model = YoloV1_Resnet18(S=7, B=2, C=20).to(device)
else:
    raise ValueError("No model selected")

# Load model checkpoint
ckpt_path = f"checkpoints/{current_model}/yolov1.pth"
assert os.path.exists(ckpt_path), "Checkpoint does not exist"
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load validation data
val_dataset = VOCDataset("val")  # Make sure this returns (img, target) but we ignore target
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Run on first image in val set
for i, (img, _) in enumerate(val_loader):
    img = img.to(device)
    with torch.no_grad():
        preds = model(img)
        bboxes = non_max_suppression(
            cellboxes_to_boxes(preds)[0],
            iou_threshold=0.5,
            threshold=0.4,
            boxformat="midpoints"
        )

    # Draw
    img_np = img[0].permute(1, 2, 0).cpu().numpy() * 255
    img_np = img_np.astype(np.uint8)
    output = draw_bounding_box(img_np, bboxes, test=True)

    # Save and break after first image
    os.makedirs("images", exist_ok=True)
    cv2.imwrite(f"images/model_output/figure_val_{i}.png", output)
    print(f"Saved: images/model_output/figure_val_{i}.png")
    
    if i >= count:
        break
