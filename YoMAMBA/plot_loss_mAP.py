import os
import torch
import matplotlib.pyplot as plt

def load_metrics(model_name):
    metrics_path = f"metrics/{model_name}/metrics.pth"
    assert os.path.exists(metrics_path), f"No saved metrics found for {model_name}"
    metrics = torch.load(metrics_path)
    train_times = []
    if model_name == "mamba":
        train_times = metrics["train_times"]
    return metrics['train_losses'], metrics['val_mAP'], train_times

# Load
mamba_train_loss, mamba_val_mAP, mamba_train_time = load_metrics("mamba")
resnet_train_loss, resnet_val_mAP, resnet_train_time = load_metrics("resnet18")

train_epochs = list(range(1, len(mamba_train_loss) + 1))
val_epochs_mamba = list(range(1, 1 + 10 * len(mamba_val_mAP), 10))
val_epochs_resnet = list(range(1, 1 + 10 * len(resnet_val_mAP), 10))

# Create plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Train loss (left y-axis)
ax1.plot(train_epochs, mamba_train_loss, label="YoMAMBA Train Loss", color='blue')
ax1.plot(train_epochs, resnet_train_loss, label="ResNet Train Loss", color='cyan')
ax1.set_xlabel("Epoch", fontsize=18)
ax1.set_ylabel("Train Loss", color='blue', fontsize=20)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.tick_params(axis='both', labelsize=18)

# Validation mAP (right y-axis)
ax2 = ax1.twinx()
ax2.plot(val_epochs_mamba, mamba_val_mAP, label="YoMAMBA Val mAP", color='red')
ax2.plot(val_epochs_resnet, resnet_val_mAP, label="ResNet Val mAP", color='orange')
ax2.set_ylabel("Val mAP", color='red', fontsize=20)
ax2.tick_params(axis='y', labelcolor='red', labelsize=18)

# YoMAMBA Train time (third y-axis)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 80))  # Offset third axis
ax3.plot(train_epochs, mamba_train_time, label="Train Time (sec)", linestyle="--", color='purple')
ax3.tick_params(axis='y', labelcolor='purple', labelsize=18)
ax3.set_ylim(0, 270)
ax3.set_yticks(range(0, 271, 30))  # ← ticks every 30 seconds

# Legend back to center left
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_3, labels_3 = ax3.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3,
           fontsize=14, loc="center left", bbox_to_anchor=(0.6, 0.45))


# Title and save
plt.title("Training Statistics", fontsize=24)
plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/figure_1.png")
plt.close()
