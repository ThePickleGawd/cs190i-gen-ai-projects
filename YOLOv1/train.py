import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import VOCDataset
from model import YOLOv1, YOLOv1ViT
from loss import YOLOLoss
import config
from utils import batch_to_mAP_list

# Make sure dirs exist
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

# Model, Optimizer, Loss
models = {
    "YOLOv1": YOLOv1,
    "YOLOv1Vit": YOLOv1ViT,
}
model = models[config.model_name]().to(config.device)
optim = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
loss_fn = YOLOLoss()

# Tracking
train_losses = []
map_scores = []
best_loss = float('inf')

# Training Loop
for epoch in range(config.EPOCHS):
    model.train()
    epoch_loss = 0

    for images, targets in train_loader:
        images, targets = images.to(config.device), targets.to(config.device)
        out = model(images)
        loss = loss_fn(out, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': avg_loss,
        }, f"checkpoints/{config.model_name}/best_model.pth")

    # Evaluation every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        metric = MeanAveragePrecision()
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
    torch.save({'losses': train_losses, 'mAP': map_scores}, f"metrics/{config.model_name}/train_metrics.pth")


# Plot loss and mAP
plt.figure()
plt.plot(train_losses, label='Loss')
plt.plot(range(0, config.EPOCHS, 5), map_scores, label='mAP')
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training Loss and mAP")
plt.legend()
plt.grid(True)
plt.savefig(f"images/{config.model_name}/metrics.png")
plt.close()
