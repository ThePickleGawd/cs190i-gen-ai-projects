import torch
import torch.nn as nn
import torchvision.models as models

from transformers import AutoModel

class YoloV1_Mamba(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloV1_Mamba, self).__init__()

        print("Using pretrained mambavision, layer4 unfrozen.")
        self.backbone = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
        # TODO: Maybe freeze some layers?

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
        out_avg_pool, features = self.backbone(x) # MAMBA supports any input resolution LOL!!! YAY
        x = features[3] # torch.Size([1, 640, 7, 7])

        x = self.yolov1head(x)
        return x
    
    def random_weight_init(self):
        for i in range(len(self.yolov1head)):
             if type(self.yolov1head[i]) == torch.nn.modules.conv.Conv2d:
                self.yolov1head[i].weight.data = self.yolov1head[i].weight.data.normal_(0, 0.02)
                self.yolov1head[i].bias.data = self.yolov1head[i].bias.data.zero_()
def test ():
    model = YoloV1_Mamba()
    x = torch.rand(2, 3, 448, 448)
    xshape = model(x).shape
    return x, xshape

testx, xdims = test()