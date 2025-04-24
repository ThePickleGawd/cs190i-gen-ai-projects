import os
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.yolov1_resnet18 import YoloV1_Resnet18
from models.yolov1_mamba import YoloV1_Mamba

from utils.yolov1_utils import get_bboxes, mean_average_precision as mAP
from data import VOCDataset
import argparse


device = "cuda"
batch_size = 64
weight_decay = 5e-4
epochs = 140
nworkers = 14 # Supposedly faster: https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
save_last_model = True
save_checkpoints = True
checkpoint_interval = 10
eval_interval = 10

mAP_train = False
mAP_val = True

parser = argparse.ArgumentParser()
parser.add_argument('--use-mamba', action='store_true', help='Use Mamba backbone instead of ResNet18')
args = parser.parse_args()

# Model selection logic
use_mamba_backbone = args.use_mamba
use_resnet18_backbone = not use_mamba_backbone
    
def main():
    # Select model
    if use_mamba_backbone:
        current_model = "mamba"
        model = YoloV1_Mamba(S=7, B=2, C=20).to(device)
    elif use_resnet18_backbone:
        current_model = "resnet18"
        model = YoloV1_Resnet18(S=7, B=2, C=20).to(device)
    else:
        print("No backbone was specified")
        return 1
    ckpt_dir = f"checkpoints/{current_model}"
    ckpt_path = f"{ckpt_dir}/resnet18_adj_lr_yolov1.cpt"

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        last_epoch = checkpoint["epoch"]
        print(f"Checkpoint from epoch:{last_epoch + 1} successfully loaded.")
        
    # Dataset
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        targets = torch.stack(targets)
        return images, targets

    train_ds = VOCDataset("train")
    val_ds = VOCDataset("val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    
    if mAP_train:
        pred_bbox, target_bbox = get_bboxes(train_loader, model, iou_threshold = 0.5, threshold = 0.4)
        train_mAP_val = mAP(pred_bbox, target_bbox, iou_threshold = 0.5, boxformat="midpoints")
        print(f"Train {train_mAP_val}")
    if mAP_val:
        val_pred_bbox, val_target_bbox = get_bboxes(val_loader, model, iou_threshold = 0.5, threshold = 0.4)
        val_mAP_val = mAP(val_pred_bbox, val_target_bbox, iou_threshold = 0.5, boxformat="midpoints")
        print(f"Val {val_mAP_val}")
            
    
            
if __name__ == "__main__":
    main()
