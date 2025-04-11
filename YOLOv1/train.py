import torchvision
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from data import VOCDataset
from torch.utils.data import DataLoader

from model import YOLO
from loss import YOLOLoss
import config

## Dataset

ds = VOCDataset()

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), torch.stack(targets)

train_dataloader = DataLoader(ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)


## Train

model = YOLO().to(config.device)
model.train()

# optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
loss_fn = YOLOLoss()

losses = []

for epoch in range(config.EPOCHS):
    epoch_loss = 0

    batch_idx = 0
    for images, targets in train_dataloader:
        images, targets = images.to(config.device), targets.to(config.device)

        out = model(images)
        loss = loss_fn(out, targets)

        # Debug exploding outputs
        print("Output stats:", out.min().item(), out.max().item(), out.mean().item())

        optim.zero_grad()
        loss.backward()

        # Debug gradient explosion
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Grad {name}: max {param.grad.abs().max().item():.4f}")

        optim.step()

        epoch_loss += loss.item()
        print(f"[Epoch {epoch+1}] Batch Loss: {loss.item():.4f}")

        batch_idx += 1
