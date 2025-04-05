import torch
import torch.nn as nn

import config

class YOLO(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []

        # Conv 1
        layers += [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # Conv 2
        layers += [
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # Conv 3
        layers += [
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # Conv 4
        for _ in range(4):
            layers += [
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1),
            ]
        layers += [
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        # Conv 5
        for _ in range(2):
            layers += [
                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1),
            ]
        layers += [
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        ]

        # Conv 6
        layers += [
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        ]

        self.model = nn.Sequential(*layers)

        self.depth = config.B * 5 + config.C
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024* config.S* config.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, config.S * config.S * self.depth)
        )

    def forward(self, X):
        X = self.model(X)
        output = self.out(X)

        return output.view(-1, config.S, config.S, self.depth)