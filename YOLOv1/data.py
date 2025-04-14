import torch
import numpy as np
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

        self.dataset = VOCDetection(root=config.DATA_PATH, download=True, year="2012", image_set=image_set, transform=transform)
        self.classes = {cls: idx for idx, cls in enumerate(config.VOC_CLASSES)}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, info = self.dataset[idx]
        
        # Predictions are encoded as a S × S × (B * (5 + C)) tensor.
        depth = config.B * (5 + config.C)
        target = torch.zeros((config.S, config.S, depth), dtype=torch.float32)

        labels = info["annotation"]["object"]
        orig_img_size = info["annotation"]["size"]
        orig_img_w, orig_img_h = int(orig_img_size["width"]), int(orig_img_size["height"])

        for label in labels:
            # Get class one-hot and the bounding box info
            name, box = label["name"], label["bndbox"]

            class_idx = self.classes[name]
            one_hot = torch.zeros(config.C)
            one_hot[class_idx] = 1

            # Scale bbox to config.IMG_SIZE (448x448)
            xmin, ymin, xmax, ymax = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
            xmin = int(box["xmin"]) * config.IMG_SIZE[0] / orig_img_w
            xmax = int(box["xmax"]) * config.IMG_SIZE[0] / orig_img_w
            ymin = int(box["ymin"]) * config.IMG_SIZE[1] / orig_img_h
            ymax = int(box["ymax"]) * config.IMG_SIZE[1] / orig_img_h

            conf = 1.0

            # Find the right slot to put it in
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            x_cell_size = config.IMG_SIZE[0] / config.S
            y_cell_size = config.IMG_SIZE[1] / config.S

            x_cell = min(int(x_center // x_cell_size), config.S - 1)
            y_cell = min(int(y_center // y_cell_size), config.S - 1)

            # Normalized x,y
            x_cell_tl = x_cell * int(x_cell_size)
            y_cell_tl = y_cell * int(y_cell_size)

            x = (x_center - x_cell_tl) / x_cell_size
            y = (y_center - y_cell_tl) / y_cell_size

            # Log scale w,h
            w = np.log((xmax - xmin) / x_cell_size)
            h = np.log((ymax - ymin) / y_cell_size)

            # Scaled w,h
            # w = (xmax - xmin) / config.IMG_SIZE[0]
            # h = (ymax - ymin) / config.IMG_SIZE[1]

            # Make box
            bbox = torch.tensor([x, y, w, h, conf])
            label_vector = torch.concat([bbox, one_hot])

            # Add to end of grid slot
            for idx in range(config.B):
                # Don't override
                if torch.any(target[y_cell, x_cell, idx*(5+config.C):(idx+1)*(5+config.C)] != 0):
                    continue

                target[y_cell, x_cell, idx*(5+config.C):(idx+1)*(5+config.C)] = label_vector
                break

        return image, target