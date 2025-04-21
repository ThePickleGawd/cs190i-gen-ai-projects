import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from loss.yolov1_loss import YoloV1Loss
from torch.optim.lr_scheduler import LambdaLR
from models.yolov1_resnet18 import YoloV1_Resnet18
from models.yolov1_mamba import YoloV1_Mamba

from utils.yolov1_utils import get_bboxes, mean_average_precision as mAP
from data import VOCDataset


device = "cuda"
batch_size = 64
weight_decay = 5e-4
epochs = 140
nworkers = 14 # Supposedly faster: https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
eval_interval = 10
save_model = True

# Select Model
use_resnet18_backbone = True
use_mamba_backbone = False

# Train

def train(train_loader, model, optimizer, loss_fn):
    """
    Input: train loader (torch loader), model (torch model), optimizer (torch optimizer)
          loss function (torch custom yolov1 loss).
    Output: loss (torch float).
    """
    # Gradient accumulation parameter: perform gradient accumulation over 16
    # batches
    accum_iter = 16
    model.train()

    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        with torch.set_grad_enabled(True):
            out = model(x)
            loss = loss_fn(out, y)
            
            loss.backward()
            total_loss += loss.item()
            
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

    return total_loss / len(train_loader)
    
def val(val_loader, model, loss_fn):
    """
    Input: val loader (torch loader), model (torch model), loss function 
          (torch custom yolov1 loss).
    Output: val loss (torch float).
    """
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss

        return total_loss / len(val_loader)

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
    else:
        print("No backbone was specified")
        return 1

    ckpt_path = f"checkpoints/{current_model}/yolov1.pth"
    metric_path = f"metrics/{current_model}/metrics.pth"
    
    # Load training settings and metrics
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    loss_fn = YoloV1Loss()

    train_loss_lst = []
    train_mAP_lst = []
    val_mAP_lst = []
    val_loss_lst = []
    last_epoch = 0

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        last_epoch = checkpoint["epoch"]
        print(f"Checkpoint from epoch:{last_epoch + 1} successfully loaded.")

    if os.path.exists(metric_path):
        m = torch.load(metric_path)
        train_loss_lst = m["train_losses"]
        train_mAP_lst = m["train_mAP"]
        val_loss_lst = m["val_losses"]
        val_mAP_lst = m["val_mAP"]


    def lr_lambda(epoch):
        if epoch <= 5: return 1 + 9 * (epoch / 5)     # linearly from 1× to 10×
        elif epoch <= 80: return 10                   # constant 10×
        elif epoch <= 110: return 1                   # back to 1×
        else: return 0.1                              # decay to 0.1×
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch if last_epoch > 0 else -1)

    # Dataset
    train_ds = VOCDataset("train")
    val_ds = VOCDataset("val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: tuple(zip(*b)))

    for epoch in tqdm(range(last_epoch, epochs)):
        # for training data
        pred_bbox, target_bbox = get_bboxes(train_loader, model, iou_threshold = 0.5, 
                                        threshold = 0.4)

        val_pred_bbox, val_target_bbox = get_bboxes(val_loader, model, iou_threshold = 0.5, 
                                        threshold = 0.4)                                
        
        # Train and val Loss
        train_loss_value = train(train_loader, model, optimizer, loss_fn)
        train_loss_lst.append(train_loss_value)
        val_loss_value = val(val_loader, model, loss_fn)
        val_loss_lst.append(val_loss_value)

        print(f"Learning Rate:", optimizer.param_groups[0]["lr"])
        print(f"Epoch:{epoch + last_epoch + 1 } Train[Loss:{train_loss_value} Val[Loss:{val_loss_value}")

        # store mAP and average mAP
        if epoch > 0 and (epoch + 1) % eval_interval == 0:
            train_mAP_val = mAP(pred_bbox, target_bbox, iou_threshold = 0.5, boxformat="midpoints")
            val_mAP_val = mAP(val_pred_bbox, val_target_bbox, iou_threshold = 0.5, boxformat="midpoints")
            train_mAP_lst.append(train_mAP_val.item())
            val_mAP_lst.append(val_mAP_val.item())
            print(f"Train mAP:{train_mAP_val}] val mAP:{val_mAP_val}]")


        if save_model == True and ( (epoch + last_epoch + 1 ) % 2) == 0 or epoch + last_epoch == epochs - 1 :
            torch.save({
                "epoch": epoch + last_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                }, ckpt_path)
            print(f"Checkpoint at {epoch + last_epoch + 1} stored")
            with open(f"results/{current_model}train_loss.txt","w") as values:
                values.write(str(train_loss_lst))
            with open(f"results/{current_model}train_mAP.txt","w") as values:
                values.write(str(train_mAP_lst))
            with open(f"results/{current_model}val_loss.txt","w") as values:
                values.write(str(val_loss_lst))
            with open(f"results/{current_model}val_mAP.txt","w") as values:
                values.write(str(val_mAP_lst))
    
            
if __name__ == "__main__":
    main()
