import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VOCDetection

import config

class VOCDataset(Dataset):
    def __init__(self, image_set="train"):
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])

        self.dataset = VOCDetection(root=config.data_path, download=False, year="2012", image_set=image_set, transform=transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Predictions are encoded as a S × S × (B ∗ 5 + C) tensor.
        depth = config.B * 5 + config.C
        target = torch.zeros((config.C, config.C, depth))



        print(image, label)

ds = VOCDataset()
print(ds[0])