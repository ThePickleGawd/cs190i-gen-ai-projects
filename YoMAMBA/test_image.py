import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from utils.yolov1_utils import non_max_suppression, cellboxes_to_boxes, draw_bounding_box
from models.yolov1_resnet18 import YoloV1_Resnet18
from models.yolov1_mamba import YoloV1_Mamba

transform = T.Compose([T.ToTensor()])
device = "cuda" if torch.cuda.is_available() else "cpu"

# Select model
use_resnet18_backbone = True
# use_resnet18_2_backbone = True
use_mamba_backbone = False

if use_mamba_backbone:
    current_model = "mamba"
    model = YoloV1_Mamba(S=7, B=2, C=20).to(device)
elif use_resnet18_backbone:
    current_model = "resnet18"
    model = YoloV1_Resnet18(S=7, B=2, C=20).to(device)
# elif use_resnet18_2_backbone:
#     current_model = "resnet18_2"
#     model = YoloV1_Resnet18(S=7, B=2, C=20).to(device)
else:
    print("No backbone was specified")
    exit(1)

# Load model
ckpt_path = f"checkpoints/{current_model}/yolov1.pth"
if not os.path.exists(ckpt_path):
    print("Checkpoint does not exist")
    exit(1)
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load and process image
image_path = 'images/sample.png'
frame = cv2.imread(image_path)
frame = cv2.resize(frame, (448, 448))
input_image = transform(frame).unsqueeze(0).to(device)

# Run detection
with torch.no_grad():
    preds = model(input_image)
    get_bboxes = cellboxes_to_boxes(preds)
    bboxes = non_max_suppression(get_bboxes[0], iou_threshold=0.5, threshold=0.4, boxformat="midpoints")

# Draw and save
output = draw_bounding_box(frame, bboxes, test=True)
cv2.imwrite("images/figure_2.png", output)

# Optionally display
# if os.environ.get('DISPLAY') is not None or os.name == 'nt':
#     cv2.imshow("Detection", output)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

print("Image processed and saved.")
