import torchvision
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from data import VOCDataset
from torch.utils.data import DataLoader

from model import YOLO
import config

## Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Dataset

ds = VOCDataset()

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), torch.stack(targets)

train_dataloader = DataLoader(ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)


## Train

model = YOLO().to(device)

for _ in range(config.EPOCHS):
    for images, targets in train_dataloader:
        images, targets = images.to(device), targets.to(device)
        print(images.shape, targets.shape)
        out = model(images)
        print(out.shape)
        break