import torch
import matplotlib.pyplot as plt
import os

def plot_metrics(train_losses, train_mAP, val_losses, val_mAP, model_name):
    os.makedirs(f"images/{model_name}", exist_ok=True)

    full_epochs = list(range(1, len(train_losses) + 1))
    sampled_epochs = list(range(10, 10 * len(val_losses) + 1, 10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(full_epochs, train_losses, label="Train Loss")
    ax1.plot(sampled_epochs, val_losses, label="Val Loss", marker='o')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} - Loss")
    ax1.legend()
    ax1.grid(True)

    # mAP plot
    ax2.plot(sampled_epochs, train_mAP, label="Train mAP", marker='o')
    ax2.plot(sampled_epochs, val_mAP, label="Val mAP", marker='o')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.set_title(f"{model_name} - mAP")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    plt.savefig(f"images/{model_name}/metrics.png")
    plt.close()

model_name = "resnet18"

# Load metrics
metrics_path = f"metrics/{model_name}/metrics.pth"
assert os.path.exists(metrics_path), "No saved metrics found."

metrics = torch.load(metrics_path)
train_losses = metrics['train_losses']
val_losses = metrics['val_losses']
train_mAP = metrics['train_mAP']
val_mAP = metrics['val_mAP']

plot_metrics(train_losses, train_mAP, val_losses, val_mAP, model_name)
print(val_mAP)
