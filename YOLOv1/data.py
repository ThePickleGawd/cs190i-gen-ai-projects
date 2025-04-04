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

        self.dataset = VOCDetection(root=config.DATA_PATH, download=False, year="2012", image_set=image_set, transform=transform)
        self.classes = {cls: idx for idx, cls in enumerate(config.VOC_CLASSES)}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, info = self.dataset[idx]
        
        # Predictions are encoded as a S × S × (B ∗ 5 + C) tensor.
        depth = config.B * 5 + config.C
        target = torch.zeros((config.C, config.C, depth), dtype=torch.float32)

        labels = info["annotation"]["object"]
        for label in labels:
            # Get class one-hot and the bounding box info
            name, box = label["name"], label["bndbox"]

            class_idx = self.classes[name]
            one_hot = torch.zeros(config.C)
            one_hot[class_idx] = 1

            xmin, ymin, xmax, ymax = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
            conf = 1.0

            # Find the right slot to put it in
            x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
            w, h = xmax - xmin, ymax - ymin
            x_cell_size, y_cell_size = config.IMG_SIZE[0] / config.S, config.IMG_SIZE[1] / config.S
            x_cell, y_cell = int(x_center // x_cell_size), int(y_center // y_cell_size)

            # Make boxes (box2 is empty obv), put it all together
            box1, box2 = torch.tensor([x_center, y_center, w, h, conf]), torch.zeros(5)
            label_vector = torch.concat([box1, box2, one_hot])

            # Insert into slot
            print(x_cell, y_cell)
            target[y_cell, x_cell, :] = label_vector

        return image, target