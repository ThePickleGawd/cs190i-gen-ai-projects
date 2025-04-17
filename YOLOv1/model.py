import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet18

import config

# Original YOLOv1 from scratch

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


# YOLOv1 with ResNet backbone

class YOLOv1ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.B * 5 + config.C

        # Load backbone ResNet
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.requires_grad_(False) # Freeze weights

        # Unfreeze top blocks so we can learn a little bit
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        self.model = nn.Sequential(
            backbone,
            Reshape(2048, 14, 14),
            DetectionNet(2048)              # 4 conv, 2 linear
        )

    def forward(self, x):
        return self.model.forward(x)

class DetectionNet(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels):
        super().__init__()

        inner_channels = 1024
        self.depth = config.B * (5 + config.C)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),   # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * inner_channels, 4096),
            # nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(4096, config.S * config.S * self.depth)
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, config.S, config.S, self.depth)
    
class Reshape(nn.Module):


# YOLOv1 with ResNet18 backbone From Scratch
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))
    
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=20):  # VOC has 20 classes
        super().__init__()
        self.base = resnet18(weights=None)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

    def forward(self, x):
        return self.features(x)

class YOLOv1ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 4096),  # fixed input dim from ResNet18
            nn.LeakyReLU(0.1),
            nn.Linear(4096, config.S * config.S * (config.B * (5 + config.C)))
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out.view(-1, config.S, config.S, config.B * (5 + config.C))
