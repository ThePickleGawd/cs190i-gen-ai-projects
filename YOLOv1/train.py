import torchvision
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from data import VOCDataset
from torch.utils.data import DataLoader

import config

## DataLoader

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), targets

ds = VOCDataset()

train_dataloader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

## Train

for _ in range(config.EPOCHS):
    for images, targets in train_dataloader:
        
        break