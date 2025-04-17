import torch
import os
from model import ResNet18Classifier, YOLOv1ResNet18
import config
from utils import plot_training_metrics, batch_to_mAP_list
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import math
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time
from data import VOCDataset
from loss import YOLOLoss

assert config.model_name == "YOLOv1ResNet18", "This script is for YOLOv1ResNet18 only"

# Create necessary directories
os.makedirs(f"checkpoints/{config.model_name}", exist_ok=True)
os.makedirs(f"images/{config.model_name}", exist_ok=True)
os.makedirs(f"metrics/{config.model_name}", exist_ok=True)

# Dataset and Dataloader
train_ds = VOCDataset("train")
test_ds = VOCDataset("val")

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), torch.stack(targets)

train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

# Load classifier weights
classifier = ResNet18Classifier().to(config.device)
resnet_checkpoint = "checkpoints/ResNet18Classifier/resnet18_voc.pth"
if os.path.exists(resnet_checkpoint):
    print(f"Loading ResNet weights from {resnet_checkpoint}")
    classifier.load_state_dict(torch.load(resnet_checkpoint, map_location=config.device))
else:
    print("Using blank ResNet weights")

# Transfer to YOLO backbone
model = YOLOv1ResNet18().to(config.device)
if os.path.exists(resnet_checkpoint):
    print(f"Transferring ResNet weights to YOLOv1 backbone")
    model.backbone.features.load_state_dict(classifier.base.state_dict(), strict=False)

loss_fn = YOLOLoss()
optim = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# Load checkpoint if exists
start_epoch = 0
best_loss = float('inf')

checkpoint_path = f"checkpoints/{config.model_name}/best_model.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    print(f"Resumed from checkpoint at epoch {start_epoch} with loss {best_loss:.4f}")

# Load saved metrics if exists
train_losses = []
map_scores = []
train_times = []

metrics_path = f"metrics/{config.model_name}/train_metrics.pth"
if os.path.exists(metrics_path):
    metrics = torch.load(metrics_path)
    train_losses = metrics['losses']
    map_scores = metrics['mAP']
    train_times = metrics['train_times']
    print(f"Loading saved metrics")

# LR Scheduler

def lr_schedule(epoch):
    if epoch < 5:
        # Warmup 1e-5
        return 0.1
    elif epoch < 15:
        # Train at 1e-4
        return 1.0
    elif epoch < 45:
        # Anneal from 1e-4 to 1e-3 over 30 epochs (cosine ramp up)
        t = (epoch - 15) / (30)
        return 10.0 * 0.5 * (1 - math.cos(math.pi * t))
    else:
        return 1.0
        # # Anneal from 1e-3 back to 1e-4 over 25 epochs (cosine decay)
        # t = (epoch - 45) / (25)
        # return 1.0 * 0.5 * (1 + math.cos(math.pi * t))

scheduler = LambdaLR(optim, lr_lambda=lr_schedule, last_epoch=start_epoch if start_epoch > 0 else -1)

first_few = 10

# Training Loop
for epoch in range(start_epoch, config.EPOCHS):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    batch_idx = 0
    for images, targets in train_loader:
        images, targets = images.to(config.device), targets.to(config.device)
        out = model(images)
        loss = loss_fn(out, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if first_few > 0:
            first_few -= 1
            print(loss.item())

        epoch_loss += loss.item()
        if batch_idx % 100 == 0:
            print(loss.item())

        batch_idx += 1

    scheduler.step()

    end_time = time.time()
    elapsed_time = end_time - start_time
    train_times.append(elapsed_time)

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': avg_loss,
        }, f"checkpoints/{config.model_name}/best_model.pth")

    # Evaluate every 10 epochs
    if epoch % 10 == 0:
        model.eval()
        metric = MeanAveragePrecision(backend="faster_coco_eval")
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(config.device), targets.to(config.device)
                preds = model(images)
                preds_list, targets_list = batch_to_mAP_list(preds, targets)
                metric.update(preds=preds_list, target=targets_list)

        result = metric.compute()
        mAP = result['map'].item()
        map_scores.append(mAP)
        print(f"[Epoch {epoch+1}] mAP: {mAP:.4f}")

    # Save metrics
    torch.save({
        'losses': train_losses,
        'mAP': map_scores,
        'train_times': train_times
    }, f"metrics/{config.model_name}/train_metrics.pth")

    plot_training_metrics(train_losses, map_scores, train_times, start_epoch, config.model_name)

