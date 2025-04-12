import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
from data import VOCDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import YOLO
from loss import YOLOLoss
import config

# Make sure dirs exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("images", exist_ok=True)

## Dataset
train_ds = VOCDataset("train")
test_ds = VOCDataset("test")

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), torch.stack(targets)

train_dataloader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
test_dataloader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

## Train
model = YOLO().to(config.device)
model.train()

optim = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
metric = MeanAveragePrecision()

loss_fn = YOLOLoss()
losses = []

for epoch in tqdm(range(config.EPOCHS)):
    model.train()
    epoch_loss = 0
    batch_idx = 0

    for images, targets in train_dataloader:
        images, targets = images.to(config.device), targets.to(config.device)

        out = model(images)
        loss = loss_fn(out, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_loss += loss.item()
        batch_idx += 1

        if batch_idx % 10 == 0:
            print(f"Batch Loss: {loss.item()}")

    avg_loss = epoch_loss / batch_idx
    losses.append(avg_loss)
    print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss:.4f}")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': avg_loss,
    }, f"checkpoints/checkpoint_epoch{epoch + 1}.pth")

    if epoch % 4 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for data, targets in tqdm(test_dataloader, desc='Test', leave=False):
                images, targets = images.to(config.device), targets.to(config.device)

                out = model(images)
                metric.update()

                # TODO: Calculate mAP

        # TODO: Save metric


torch.save({
    'model_state_dict': model.state_dict(),
}, "checkpoints/model.pth")

# Plot loss vs epoch
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Loss vs Epoch")
plt.grid(True)
plt.savefig("images/loss_vs_epoch.png")
