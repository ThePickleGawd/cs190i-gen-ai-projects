import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.datasets import VOCDetection
from torchvision.tv_tensors import BoundingBoxes, Image
import torchvision

import config

class VOCDataset(Dataset):
    def __init__(self, image_set="train"):
        self.is_train = image_set == "train"

        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(config.IMG_SIZE),
            v2.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2)
            ) if self.is_train else v2.Identity(),
            v2.ColorJitter(saturation=0.5, brightness=0.5) if self.is_train else v2.Identity(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.dataset = VOCDetection(root=config.DATA_PATH, download=False, year="2012", image_set=image_set, transform=None)
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

        image = Image(image)
        boxes = BoundingBoxes(boxes, format="XYXY", canvas_size=config.IMG_SIZE)

        image, boxes = self.image_transform(image, boxes)

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
            label_vector = torch.concat([bbox, one_hot])

            for b in range(config.B):
                if torch.any(target[y_cell, x_cell, b*(5+config.C):(b+1)*(5+config.C)] != 0):
                    continue
                target[y_cell, x_cell, b*(5+config.C):(b+1)*(5+config.C)] = label_vector
                break

        return image, target


class VOCClassificationDataset(Dataset):
    def __init__(self, image_set="train"):
        # load VOC
        self.voc = VOCDetection(
            root=config.DATA_PATH,
            year="2012",
            image_set=image_set,
            download=False,
            transform=None
        )
        # mapping class name => index
        self.classes = config.VOC_CLASSES
        self.class_to_idx = {cls:i for i,cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(config.IMG_SIZE),
            v2.RandomHorizontalFlip() if image_set=="train" else v2.Identity(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        pil_img, info = self.voc[idx]
        # pull out all object labels (might be a single dict)
        labels = info["annotation"]["object"]
        if not isinstance(labels, list):
            labels = [labels]

        # build multi‑hot target
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for obj in labels:
            cls_name = obj["name"]
            y[self.class_to_idx[cls_name]] = 1.0

        # apply your v2 transforms
        img = Image(pil_img)
        img = self.image_transform(img)

        return img, y
