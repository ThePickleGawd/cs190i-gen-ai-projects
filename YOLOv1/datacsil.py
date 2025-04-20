import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2, functional as F
from torchvision.datasets import VOCDetection
from PIL import Image as PILImage

import config

class VOCDataset(Dataset):
    def __init__(self, image_set="train"):
        self.is_train = image_set == "train"

        self.image_transform = v2.Compose([
            v2.Resize(config.IMG_SIZE),
            v2.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2)
            ) if self.is_train else v2.Identity(),
            v2.ColorJitter(saturation=0.5, brightness=0.5) if self.is_train else v2.Identity(),
            v2.ToTensor(),  # replaces ToImage + ToDtype
        ])

        self.dataset = VOCDetection(root=config.DATA_PATH, download=False, year="2012", image_set=image_set)
        self.classes = {cls: idx for idx, cls in enumerate(config.VOC_CLASSES)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, info = self.dataset[idx]

        labels = info["annotation"]["object"]
        orig_img_size = info["annotation"]["size"]
        orig_img_w, orig_img_h = int(orig_img_size["width"]), int(orig_img_size["height"])

        if not isinstance(labels, list):
            labels = [labels]

        boxes = []
        class_ids = []

        for label in labels:
            box = label["bndbox"]
            xmin = int(box["xmin"]) * config.IMG_SIZE[0] / orig_img_w
            xmax = int(box["xmax"]) * config.IMG_SIZE[0] / orig_img_w
            ymin = int(box["ymin"]) * config.IMG_SIZE[1] / orig_img_h
            ymax = int(box["ymax"]) * config.IMG_SIZE[1] / orig_img_h
            boxes.append([xmin, ymin, xmax, ymax])
            class_ids.append(self.classes[label["name"]])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.int64)

        image = self.image_transform(PILImage.fromarray(np.array(image)))

        # Construct target tensor
        depth = config.B * (5 + config.C)
        target = torch.zeros((config.S, config.S, depth), dtype=torch.float32)

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            class_idx = class_ids[i]
            one_hot = torch.zeros(config.C)
            one_hot[class_idx] = 1

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            x_cell_size = config.IMG_SIZE[0] / config.S
            y_cell_size = config.IMG_SIZE[1] / config.S

            x_cell = min(int(x_center // x_cell_size), config.S - 1)
            y_cell = min(int(y_center // y_cell_size), config.S - 1)

            x_cell_tl = x_cell * int(x_cell_size)
            y_cell_tl = y_cell * int(y_cell_size)

            x = (x_center - x_cell_tl) / x_cell_size
            y = (y_center - y_cell_tl) / y_cell_size
            w = (xmax - xmin) / config.IMG_SIZE[0]
            h = (ymax - ymin) / config.IMG_SIZE[1]

            bbox = torch.tensor([x, y, w, h, 1.0])
            label_vector = torch.cat([bbox, one_hot])

            for b in range(config.B):
                start = b * (5 + config.C)
                end = (b + 1) * (5 + config.C)
                if torch.any(target[y_cell, x_cell, start:end] != 0):
                    continue
                target[y_cell, x_cell, start:end] = label_vector
                break

        return image, target
