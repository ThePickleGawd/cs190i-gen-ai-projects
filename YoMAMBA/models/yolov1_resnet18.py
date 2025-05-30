import ssl
import urllib.request

# Override SSL context globally for downloads
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn

import torchvision.models as models


class YoloV1_Resnet18(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloV1_Resnet18, self).__init__()

        print("Using pretrained resnet18, layer4 unfrozen.")
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Freeze all layers
        for param in resnet.parameters():
            param.requires_grad = False
        # Unfreeze layer4 (last conv block)
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.resnet18backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.yolov1head = nn.Sequential (
            # Block 5 (last two conv layers)
            # Since the last ResNet 18 layer consists of a (3x3, 512) conv layer
            # we adjust the input size of the yolo head from 1024 to 512.
            nn.Conv2d(in_channels = 512, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 2,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
           
            # Block 6
            nn.Conv2d(in_channels = 1024, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            # prediction block
            nn.Flatten(),
            nn.Linear(in_features = 1024 * S * S, out_features = 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features = 4096, out_features = S * S * (C + B * 5)),
            # reshape in loss to be (S, S, 30) with C + B * 5 = 30
            )

        self.random_weight_init()

    def forward(self, x):
        x = self.resnet18backbone(x) # (N, 512, 14, 14)
        x = self.yolov1head(x)
        return x
    
    def random_weight_init(self):
        for i in range(len(self.yolov1head)):
             if type(self.yolov1head[i]) == torch.nn.modules.conv.Conv2d:
                self.yolov1head[i].weight.data = self.yolov1head[i].weight.data.normal_(0, 0.02)
                self.yolov1head[i].bias.data = self.yolov1head[i].bias.data.zero_()