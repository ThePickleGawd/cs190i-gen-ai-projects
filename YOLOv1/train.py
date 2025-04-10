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

        with open('log.txt', 'a') as f:
            print(out.flatten(), file=f)

        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_loss += loss.item()
        print(f"[Epoch {epoch+1}] Batch Loss: {loss.item():.4f}")

        batch_idx += 1

    avg_loss = epoch_loss / len(train_dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    break

    # Save model checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': avg_loss,
    }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')


torch.save(model.state_dict(), f"checkpoints/model.pth")