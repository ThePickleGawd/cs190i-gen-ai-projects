#!/usr/bin/env python3
import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data import VOCClassificationDataset  # Use multi-label dataset
from model import ResNet18
import config
from utils import plot_training_metrics

def main():
    parser = argparse.ArgumentParser("ResNet18 Classification Training")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-last-checkpoint", action="store_true", default=False)
    args = parser.parse_args()

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}, model: ResNet18")

    os.makedirs("checkpoints/ResNet18", exist_ok=True)
    os.makedirs("metrics/ResNet18", exist_ok=True)
    os.makedirs("images/ResNet18", exist_ok=True)

    train_ds = VOCClassificationDataset("train")
    val_ds = VOCClassificationDataset("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ResNet18().to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float("inf")
    ckpt_path = "checkpoints/ResNet18/last_model.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_loss = ckpt["loss"]
        print(f"Resumed from epoch {start_epoch}, loss {best_loss:.4f}")

    train_losses, train_times = [], []
    metrics_path = "metrics/ResNet18/train_metrics.pth"
    if os.path.exists(metrics_path):
        m = torch.load(metrics_path)
        train_losses = m["losses"][:start_epoch]
        train_times  = m["times"][:start_epoch]
        print("Loaded previous metrics.")

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_times.append(elapsed)

        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, ckpt_path)

        if args.save_last_checkpoint:
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, "checkpoints/ResNet18/last_model.pth")

        torch.save({
            "losses": train_losses,
            "times": train_times
        }, metrics_path)

        plot_training_metrics(train_losses, [], train_times, model_name="ResNet18")


if __name__ == "__main__":
    main()
