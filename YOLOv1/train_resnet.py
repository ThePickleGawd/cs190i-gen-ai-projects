import torch
from model import ResNet18Classifier, YOLOv1ResNet18
from loss import ResNetLoss

# Load classifier weights
classifier = ResNet18Classifier()
loss = ResNetLoss()



