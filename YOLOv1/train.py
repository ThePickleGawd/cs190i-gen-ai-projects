#!/usr/bin/env python3
import argparse
import os
import time
import math
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from data import VOCDataset
from model import YOLOv1, YOLOv1ViT, YOLOv1ResNet
from loss import YOLOLoss
import config
from utils import batch_to_mAP_list, plot_training_metrics


def main():
    # CLI arguments
    parser = argparse.ArgumentParser("YOLO Training")
    parser.add_argument("--model", choices=["YOLOv1","YOLOv1ViT","YOLOv1ResNet"], default="YOLOv1ResNet")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--save-last-checkpoint", action="store_true", default=False)
    args = parser.parse_args()

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}, model: {args.model}")

    # Dirs
    os.makedirs(f"checkpoints/{args.model}", exist_ok=True)
    os.makedirs(f"metrics/{args.model}", exist_ok=True)
    os.makedirs(f"images/{args.model}", exist_ok=True)

    # Data
    train_ds = VOCDataset("train")
    val_ds = VOCDataset("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: tuple(zip(*b)))

    # Model
    model_cls = {"YOLOv1": YOLOv1, "YOLOv1ViT": YOLOv1ViT, "YOLOv1ResNet": YOLOv1ResNet}[args.model]
    model = model_cls().to(device)

    # Optimizer with parameter groups for ResNet fine-tuning
    if args.model == "YOLOv1ResNet":
        backbone = model.model[0]
        backbone.requires_grad_(False)
        for layer in (backbone.layer3, backbone.layer4):
            for p in layer.parameters():
                p.requires_grad = True
        detector = model.model[2]
        optimizer = SGD([
            {"params": backbone.layer3.parameters(), "lr": args.lr * 0.1},
            {"params": backbone.layer4.parameters(), "lr": args.lr * 0.1},
            {"params": detector.parameters(),     "lr": args.lr},
        ], momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Loss
    loss_fn = YOLOLoss(lambda_class=args.lambda_cls)

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    ckpt_path = f"checkpoints/{args.model}/last_model.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_loss = ckpt['loss']
        print(f"Resumed from epoch {start_epoch}, loss {best_loss:.4f}")

    # Metrics history
    train_losses, map_scores, train_times = [], [], []
    metrics_path = f"metrics/{args.model}/train_metrics.pth"
    if os.path.exists(metrics_path):
        m = torch.load(metrics_path)
        train_losses = m['losses'][:start_epoch]
        map_scores   = m['mAP'][: start_epoch // config.EVAL_INTERVAL]
        train_times  = m['times'][:start_epoch]
        print("Loaded previous metrics.")

    # LR Scheduler
    def lr_lambda(epoch):
        if epoch < 25:
            return 1.0
        elif epoch < 75:
            t = (epoch - 25) / 50
            return 10.0 * 0.5 * (1 - math.cos(math.pi * t))
        else:
            return 1.0
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=start_epoch-1)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, tgts in pbar:
            imgs = torch.stack(imgs).to(device)
            tgts = torch.stack(tgts).to(device)
            preds = model(imgs)
            loss = loss_fn(preds, tgts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        elapsed = time.time() - t0
        train_times.append(elapsed)
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, ckpt_path)

        # Save last
        if args.save_last_checkpoint:
            last_path = f"checkpoints/{args.model}/last_model.pth"
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, last_path)

        # Evaluate mAP
        if (epoch % config.EVAL_INTERVAL) == 0:
            model.eval()
            metric = MeanAveragePrecision(backend="faster_coco_eval")
            with torch.no_grad():
                for imgs, tgts in val_loader:
                    imgs = torch.stack(imgs).to(device)
                    tgts = torch.stack(tgts).to(device)
                    preds = model(imgs)
                    p_list, t_list = batch_to_mAP_list(preds, tgts)
                    metric.update(preds=p_list, target=t_list)
            mAP = metric.compute()['map'].item()
            map_scores.append(mAP)
            print(f"[Epoch {epoch+1}] mAP: {mAP:.4f}")

        # Save metrics
        torch.save({
            'losses': train_losses,
            'mAP': map_scores,
            'times': train_times
        }, metrics_path)

        # Plot metrics
        plot_training_metrics(train_losses, map_scores, train_times, args.model)


if __name__ == '__main__':
    main()
