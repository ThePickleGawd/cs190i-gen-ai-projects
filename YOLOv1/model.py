import torch
import torch.nn as nn
import torch.nn.functional as F

import config

class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        # Conv 1
        layers += [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # Conv 2
        layers += [
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # Conv 3
        layers += [
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # Conv 4
        for _ in range(4):
            layers += [
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
            ]
        layers += [
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # Conv 5
        for _ in range(2):
            layers += [
                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
            ]
        layers += [
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        ]

        # Conv 6
        for _ in range(2):
            layers += [
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
            ]

        self.model = nn.Sequential(*layers)

        self.depth = config.B * (5 + config.C)
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * config.S * config.S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, config.S * config.S * self.depth),
        )
 
    def forward(self, X):
        X = self.model(X)
        output = self.out(X)

        # x,y,conf sigmoid for each bounding box
        output = output.view(-1, config.S, config.S, config.B, 5 + config.C)
        output[..., 0:2] = F.sigmoid(output[..., 0:2])
        output[..., 4] = F.sigmoid(output[..., 4])

        output = output.view(-1, config.S, config.S, self.depth)
        return output
    
class YOLOv1ViT(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, X):
        pass