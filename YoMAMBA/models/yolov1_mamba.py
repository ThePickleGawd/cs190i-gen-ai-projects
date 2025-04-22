import torch
import torch.nn as nn
import torchvision.models as models

from transformers import AutoModel

class YoloV1_Mamba(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloV1_Mamba, self).__init__()

        print("Using pretrained mambavision, weights all frozen")
        mambavision = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
        for param in mambavision.parameters():
            param.requires_grad = False

        self.backbone = mambavision
        

        self.yolov1head = nn.Sequential(
            nn.Conv2d(640, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # 14x14 → 7x7
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),
        )


        self.random_weight_init()

    def forward(self, x):
        out_avg_pool, features = self.backbone(x) # MAMBA supports any input resolution LOL!!! YAY
        x = features[3] # torch.Size([N, 640, 14, 14])
        x = self.yolov1head(x)
        return x
    
    def random_weight_init(self):
        for i in range(len(self.yolov1head)):
             if type(self.yolov1head[i]) == torch.nn.modules.conv.Conv2d:
                self.yolov1head[i].weight.data = self.yolov1head[i].weight.data.normal_(0, 0.02)
                self.yolov1head[i].bias.data = self.yolov1head[i].bias.data.zero_()