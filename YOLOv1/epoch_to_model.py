import torch
from model import YOLOv1

checkpoint = torch.load("checkpoints/checkpoint_epoch81.pth")
model = YOLOv1()
model.load_state_dict(checkpoint['model_state_dict'])
torch.save(model.state_dict(), "checkpoints/model.pth")
