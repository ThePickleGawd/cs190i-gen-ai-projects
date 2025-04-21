import torch
import matplotlib.pyplot as plt
import os
from utils import plot_training_metrics


model_name = "YOLOv2ResNet"

# Load metrics
metrics_path = f"metrics/{model_name}/train_metrics.pth"
assert os.path.exists(metrics_path), "No saved metrics found."

metrics = torch.load(metrics_path)
train_losses = metrics['losses']
map_scores = metrics['mAP']
train_times = metrics['times']
plot_training_metrics(train_losses, map_scores, train_times, model_name)
print(map_scores)