import torchvision
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from data import VOCDataset
from torch.utils.data import DataLoader

from model import YOLO
from loss import YOLOLoss
import config

## Config

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

## Dataset

ds = VOCDataset()

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), torch.stack(targets)

train_dataloader = DataLoader(ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)


## Train

model = YOLO().to(device)
model.train()

optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
loss_fn = YOLOLoss()

for _ in range(config.EPOCHS):
    for images, targets in train_dataloader:
        images, targets = images.to(device), targets.to(device)

        out = model(images)

        loss = loss_fn(out, targets)
        
        optim.zero_grad()
        loss.backwards()
        optim.step()
        break
        