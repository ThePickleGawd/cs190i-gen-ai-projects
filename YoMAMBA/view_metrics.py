import torch
import matplotlib.pyplot as plt
import os
from utils import plot_training_metrics

"""
torch.save({
                "train_losses": train_loss_list,
                "train_mAP": train_mAP_list,
                "val_losses": val_loss_list,
                "val_mAP": val_mAP_list
            }, metric_path)
"""


model_name = "resnet18"

# Load metrics
metrics_path = f"metrics/{model_name}/metrics.pth"
assert os.path.exists(metrics_path), "No saved metrics found."

metrics = torch.load(metrics_path)
train_losses = metrics['losses']
map_scores = metrics['mAP']
train_times = metrics['times']
plot_training_metrics(train_losses, map_scores, train_times, model_name)
print(map_scores)