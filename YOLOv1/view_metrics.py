import torch
import matplotlib.pyplot as plt
import os
import config  # assumes config.model_name and config.EPOCHS are defined

# Load metrics
metrics_path = f"metrics/{config.model_name}/train_metrics.pth"
assert os.path.exists(metrics_path), "No saved metrics found."

metrics = torch.load(metrics_path)
train_losses = metrics['losses']
map_scores = metrics['mAP']
train_times = metrics['train_times']

# Plot Loss and mAP
plt.figure()
plt.plot(train_losses, label='Loss')
plt.plot(range(0, len(train_losses), 5), map_scores, label='mAP')  # assumes evaluation every 5 epochs
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training Loss and mAP")
plt.legend()
plt.grid(True)
plt.savefig(f"images/{config.model_name}/metrics.png")
plt.show()

# Plot training time
plt.figure()
plt.plot(train_times, label='Train Time (s)')
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.title("Training Time per Epoch")
plt.grid(True)
plt.savefig(f"images/{config.model_name}/train_times.png")
plt.show()
