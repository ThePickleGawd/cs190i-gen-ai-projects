import torchvision
import torch
from torch.utils.data import DataLoader


ds_train = torchvision.datasets.VOCDetection(root="data", download=True, year="2012", image_set="train")
train_dataloader = DataLoader(ds_train, batch_size=32)

print(ds_train[0])