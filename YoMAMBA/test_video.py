import os
import cv2
import numpy as np
import time 
import torch
import torch.optim as optim
from utils.yolov1_utils import non_max_suppression, cellboxes_to_boxes, get_bboxes
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from models.yolov1_resnet18 import YoloV1_Resnet18
from models.yolov1_mamba import YoloV1_Mamba
import matplotlib.pyplot as plt
from utils.yolov1_utils import draw_bounding_box

transform = T.Compose([T.ToTensor()])
weight_decay = 0.0005
device = "cuda" if torch.cuda.is_available() else "cpu"

# Select model
use_resnet18_backbone = True
use_mamba_backbone = False

if use_mamba_backbone:
    current_model = "mamba"
    model = YoloV1_Mamba(S=7, B=2, C=20).to(device)
elif use_resnet18_backbone:
    current_model = "resnet18"
    model = YoloV1_Resnet18(S=7, B=2, C=20).to(device)
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

# video captioning
video_path = 'video/sample_video.mp4'
cap = cv2.VideoCapture(video_path)

fps = 0
fps_start = 0
prev = 0 
video_rec = cv2.VideoWriter(f'video/yolo_output.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30, (448, 448))

def can_use_imshow():
    return os.environ.get('DISPLAY') is not None or os.name == 'nt'

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.array(frame)
    frame = cv2.resize(frame, (448, 448))
    input_frame = cv2.resize(frame, (448, 448))
    fps_end = time.time() 
    time_diff = fps_end - fps_start
    fps = int(1 / (time_diff - prev))
    prev = fps_end
    height, width = frame.shape[:2]
    fps_txt = "FPS: {}".format(fps)
    frame = cv2.putText(frame, fps_txt, (width - 90, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

    input_frame = transform(input_frame)
    input_frame = input_frame.unsqueeze(0)
    preds = model(input_frame.to(device))
    
    get_bboxes = cellboxes_to_boxes(preds)
    # Note: Predictions are made on the image tensor. Pixel values are automaticaly normalised to range 
    # [0,1]. It is possibel to convert the image back via 1. input.frame.squeeze(0) and 
    # 2. frame.permute(1, 2, 0).numpy() * 255. This is however not necessary, since we keep a copy of the
    # original non-normalised frame, which we draw bounding boxes on instead of the normalized torch tensor image. 
    bboxes = non_max_suppression(get_bboxes[0], iou_threshold = 0.5, threshold = 0.4, boxformat = "midpoints")
    frame = draw_bounding_box(frame, bboxes, test = True)
    
    if can_use_imshow():
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_rec.write(frame)

video_rec.release()
cap.release()
cv2.destroyAllWindows()
print("Video processed in real time and saved.")
