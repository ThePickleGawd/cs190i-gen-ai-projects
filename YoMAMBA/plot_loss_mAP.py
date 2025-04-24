import os
import torch
import matplotlib.pyplot as plt

def load_metrics(model_name):
    metrics_path = f"metrics/{model_name}/metrics.pth"
    assert os.path.exists(metrics_path), f"No saved metrics found for {model_name}"
    metrics = torch.load(metrics_path)
    return metrics['train_losses'], metrics['val_mAP']

# Load
mamba_train_loss, mamba_val_mAP = load_metrics("mamba")
resnet_train_loss, resnet_val_mAP = load_metrics("resnet18")
# resnet_no_augment_train_loss, resnet_no_augment_val_mAP = load_metrics("resnet101-no-augment")
# resnet_no_augment_val_mAP = [0] + resnet_no_augment_val_mAP

train_epochs = list(range(1, len(mamba_train_loss) + 1))
val_epochs_mamba = list(range(1, 1 + 10 * len(mamba_val_mAP), 10))
val_epochs_resnet = list(range(1, 1 + 10 * len(resnet_val_mAP), 10))
val_no_augment_epochs_resnet = list(range(1, 1 + 10 * len(resnet_val_mAP), 10))

# Create plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot train losses on left y-axis
ax1.plot(train_epochs, mamba_train_loss, label="YoMAMBA Train Loss", color='blue')
ax1.plot(train_epochs, resnet_train_loss, label="ResNet Train Loss", color='cyan')
# ax1.plot(train_epochs, resnet_no_augment_train_loss, label="ResNet* Train Loss", color='green')
ax1.set_xlabel("Epoch", fontsize=18)
ax1.set_ylabel("Train Loss", color='blue', fontsize=20)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.tick_params(axis='both', labelsize=18)

# Create second y-axis for val mAP
ax2 = ax1.twinx()
ax2.plot(val_epochs_mamba, mamba_val_mAP, label="YoMAMBA Val mAP", color='red')
ax2.plot(val_epochs_resnet, resnet_val_mAP, label="ResNet Val mAP", color='orange')
# ax2.plot(val_no_augment_epochs_resnet, resnet_no_augment_val_mAP, label="ResNet* Val mAP", color='green')
ax2.set_ylabel("Val mAP", color='red', fontsize=20)
ax2.tick_params(axis='y', labelcolor='red', labelsize=18)

# Legend handling
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=16,
           loc="center left", bbox_to_anchor=(0.6, 0.45))

plt.title("Training @ 223.7 sec/epoch", fontsize=24)
plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/figure_1.png")
plt.close()
