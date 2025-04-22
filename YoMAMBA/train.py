import os
import torch
import time
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from loss.yolov1_loss import YoloV1Loss
from torch.optim.lr_scheduler import LambdaLR
from models.yolov1_resnet18 import YoloV1_Resnet18
from models.yolov1_resnet101 import YoloV1_Resnet101
from models.yolov1_mamba import YoloV1_Mamba

from utils.yolov1_utils import get_bboxes, mean_average_precision as mAP
from data import VOCDataset


device = "cuda"
batch_size = 64
weight_decay = 5e-4
epochs = 140
nworkers = 14 # Supposedly faster: https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
save_last_model = True
save_checkpoints = True
checkpoint_interval = 10
eval_interval = 10

# Select Model
use_resnet18_backbone = False
use_resnet101_backbone = False
use_mamba_backbone = True

# Train Model
def train(train_loader, model, optimizer, loss_fn, scheduler, epoch):
    """
    Input: train loader (torch loader), model (torch model), optimizer (torch optimizer)
          loss function (torch custom yolov1 loss).
    Output: loss (torch float).
    """
    model.train()

    total_loss = 0.0
    t0 = time.time()
    pbar = tqdm(train_loader, desc=f"Train: Epoch {epoch+1}/{epochs}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        out = model(x)
        loss = loss_fn(out, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    scheduler.step()
    elapsed = time.time() - t0
    avg_loss = total_loss / len(train_loader)
    return avg_loss, elapsed
    
def val(val_loader, model, loss_fn, epoch):
    """
    Input: val loader (torch loader), model (torch model), loss function 
          (torch custom yolov1 loss).
    Output: val loss (torch float).
    """
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        t0 = time.time()
        pbar = tqdm(val_loader, desc=f"Val: Epoch: {epoch+1}/{epochs}")
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y)

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        elapsed = time.time() - t0
        avg_loss = total_loss / len(val_loader)
        return avg_loss, elapsed
    
def main():
    # Select model
    if use_mamba_backbone:
        lr = 1e-5
        current_model = "mamba"
        model = YoloV1_Mamba(S=7, B=2, C=20).to(device)
    elif use_resnet18_backbone:
        lr =  1e-5
        current_model = "resnet18"
        model = YoloV1_Resnet18(S=7, B=2, C=20).to(device)
    elif use_resnet101_backbone:
        lr =  1e-5
        current_model = "resnet101"
        model = YoloV1_Resnet101(S=7, B=2, C=20).to(device)
    else:
        print("No backbone was specified")
        return 1

    ckpt_dir = f"checkpoints/{current_model}"
    metric_dir = f"metrics/{current_model}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/yolov1.pth"
    metric_path = f"{metric_dir}/metrics.pth"
    
    # Load training settings and metrics
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    loss_fn = YoloV1Loss()

    train_loss_list = []
    train_mAP_list = []
    train_times_list = []
    val_mAP_list = []
    val_loss_list = []
    val_times_list = []
    last_epoch = 0

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        last_epoch = checkpoint["epoch"]
        print(f"Checkpoint from epoch:{last_epoch + 1} successfully loaded.")

    if os.path.exists(metric_path):
        m = torch.load(metric_path)
        train_loss_list = m["train_losses"]
        train_mAP_list = m["train_mAP"]
        train_times_list = m["train_times"]
        val_loss_list = m["val_losses"]
        val_mAP_list = m["val_mAP"]
        val_times_list = m["val_times"]

    def lr_lambda(epoch):
        if epoch <= 5: return 1 + 9 * (epoch / 5)     # linearly from 1× to 10×
        elif epoch <= 80: return 10                   # constant 10×
        elif epoch <= 110: return 1                   # back to 1×
        else: return 0.1                              # decay to 0.1×
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch if last_epoch > 0 else -1)

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

    for epoch in range(last_epoch, epochs):
        
        # Train Step
        train_loss_value, train_time = train(train_loader, model, optimizer, loss_fn, scheduler, epoch)
        train_loss_list.append(train_loss_value)
        train_times_list.append(train_time)

        print(
            f"Epoch {epoch + 1} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Train Loss: {train_loss_value:.4f} ({train_time:.2f}s)"
        )

        # Evaluate: Val loss, train mAP, val mAP
        if epoch == 0 or (epoch + 1) % eval_interval == 0:
            val_loss_value, val_time = val(val_loader, model, loss_fn, epoch)
            val_loss_list.append(val_loss_value)
            val_times_list.append(val_time)
            pred_bbox, target_bbox = get_bboxes(train_loader, model, iou_threshold = 0.5, threshold = 0.4)
            val_pred_bbox, val_target_bbox = get_bboxes(val_loader, model, iou_threshold = 0.5, threshold = 0.4)                                
            train_mAP_val = mAP(pred_bbox, target_bbox, iou_threshold = 0.5, boxformat="midpoints")
            val_mAP_val = mAP(val_pred_bbox, val_target_bbox, iou_threshold = 0.5, boxformat="midpoints")
            train_mAP_list.append(train_mAP_val.item())
            val_mAP_list.append(val_mAP_val.item())
            
            print(
                f"Val Loss: {val_loss_value:.4f} ({val_time:.2f}s) | "
                f"Train mAP: {train_mAP_val:.4f} | Val mAP: {val_mAP_val:.4f}"
            )

        if save_last_model:
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, ckpt_path)
            torch.save({
                "train_losses": train_loss_list,
                "train_mAP": train_mAP_list,
                "train_times": train_times_list,
                "val_losses": val_loss_list,
                "val_mAP": val_mAP_list,
                "val_times": val_times_list
            }, metric_path)
            print(f"Saved last model and metrics")

        if save_checkpoints and (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"))
            print(f"Checkpoint at {epoch + 1} stored")
            
    
            
if __name__ == "__main__":
    main()
