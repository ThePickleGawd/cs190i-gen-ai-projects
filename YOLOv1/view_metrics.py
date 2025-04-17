import torch
import matplotlib.pyplot as plt
import os
import config  # assumes config.model_name and config.EPOCHS are defined
from utils import plot_training_metrics


# Load metrics
metrics_path = f"metrics/YOLOv1ResNet/train_metrics.pth"
assert os.path.exists(metrics_path), "No saved metrics found."

metrics = torch.load(metrics_path)
train_losses = metrics['losses']
map_scores = metrics['mAP']
train_times = metrics['train_times']
start_epoch = 0  # or load from checkpoint if resuming
plot_training_metrics(train_losses, map_scores, train_times, start_epoch, config.model_name)
print(map_scores)