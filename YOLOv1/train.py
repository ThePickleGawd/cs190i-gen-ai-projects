import argparse
import os
import time
import math

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from data import VOCDataset
from model import YOLOv1, YOLOv1ResNet, YOLOv1ViT
from loss import YOLOLoss
from utils import batch_to_mAP_list, plot_training_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO variants on PASCAL VOC")
    parser.add_argument("--model", required=True,
                        choices=["YOLOv1", "YOLOv1ResNet", "YOLOv1ViT"],
                        help="Which model to train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda-cls", type=float, default=1.0,
                        help="Classification loss weight")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Epochs between mAP evaluations")
    parser.add_argument("--save-dir", type=str, default=".")
    args = parser.parse_args()

    # Dirs
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints', args.model)
    metrics_dir = os.path.join(args.save_dir, 'metrics', args.model)
    images_dir = os.path.join(args.save_dir, 'images', args.model)
    for d in (ckpt_dir, metrics_dir, images_dir):
        os.makedirs(d, exist_ok=True)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # Data
    train_ds = VOCDataset('train')
    val_ds   = VOCDataset('val')
    def collate_fn(batch):
        imgs, targets = zip(*batch)
        return torch.stack(imgs), torch.stack(targets)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    # Model
    model_map = {
        'YOLOv1': YOLOv1,
        'YOLOv1ResNet': YOLOv1ResNet,
        'YOLOv1ViT': YOLOv1ViT
    }
    model = model_map[args.model]().to(device)

    # Freeze/unfreeze for ResNet variant
    if args.model == 'YOLOv1ResNet':
        backbone = model.model[0]
        backbone.requires_grad_(False)
        for layer in (backbone.layer3, backbone.layer4):
            for p in layer.parameters():
                p.requires_grad = True

    # Optimizer
    if args.model == 'YOLOv1ResNet':
        detector = model.model[2]
        optimizer = SGD([
            {'params': backbone.layer3.parameters(), 'lr': args.lr * 0.1},
            {'params': backbone.layer4.parameters(), 'lr': args.lr * 0.1},
            {'params': detector.parameters(),         'lr': args.lr},
        ], momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)

    # Loss & Scheduler
    loss_fn = YOLOLoss(lambda_class=args.lambda_cls)
    def lr_schedule(epoch):
        if epoch < 15:
            return 1.0
        elif epoch < 45:
            t = (epoch - 15) / 30
            return 10.0 * 0.5 * (1 - math.cos(math.pi * t))
        else:
            return 1.0
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    # Checkpoint resume
    start_epoch = 0
    best_loss = float('inf')
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
    last_path = os.path.join(ckpt_dir, 'last_model.pth')
    if os.path.exists(ckpt_path):
        cp = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        start_epoch = cp['epoch']
        best_loss = cp['loss']
        print(f"Resumed {args.model} at epoch {start_epoch}, loss {best_loss:.4f}")

    # Metrics resume
    metrics_path = os.path.join(metrics_dir, 'train_metrics.pth')
    train_losses, map_scores, train_times = [], [], []
    if os.path.exists(metrics_path):
        data = torch.load(metrics_path)
        train_losses = data['losses'][:start_epoch]
        map_scores   = data['mAP'][:start_epoch // args.eval_interval]
        train_times  = data['train_times'][:start_epoch]
        print("Loaded previous metrics")

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for imgs, targets in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        elapsed = time.time() - t0
        train_times.append(elapsed)
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, ckpt_path)

        # Save last model every epoch
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, last_path)

        # Evaluate mAP
        if (epoch+1) % args.eval_interval == 0:
            model.eval()
            metric = MeanAveragePrecision(backend="faster_coco_eval")
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    preds = model(imgs)
                    p_list, t_list = batch_to_mAP_list(preds, targets)
                    metric.update(preds=p_list, target=t_list)
            mAP = metric.compute()['map'].item()
            map_scores.append(mAP)
            print(f"[Epoch {epoch+1}] mAP: {mAP:.4f}")

        # Save metrics & plot
        torch.save({
            'losses': train_losses,
            'mAP': map_scores,
            'train_times': train_times
        }, metrics_path)
        plot_training_metrics(train_losses, map_scores, train_times, start_epoch, args.model)
