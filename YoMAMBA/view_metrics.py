import torch
import matplotlib.pyplot as plt
import os

def plot_metrics(train_losses, train_mAP, val_losses, val_mAP, train_times, model_name):
    os.makedirs(f"images/{model_name}", exist_ok=True)

    full_epochs = list(range(1, len(train_losses) + 1))
    sampled_epochs = [1] + full_epochs[::10]

    fig, ax1 = plt.subplots()

    # Plot Loss
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax1.plot(full_epochs, train_losses, label="Train Loss", color='tab:blue')
    ax1.plot(sampled_epochs, val_losses, label="Val Loss", linestyle='--', color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # mAP on second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("mAP", color='tab:red')
    ax2.plot(sampled_epochs, train_mAP, label="Train mAP", color='tab:red')
    ax2.plot(sampled_epochs, val_mAP, label="Val mAP", linestyle='--', color='tab:pink')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Training time on third y-axis
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("outward", 60))
    ax3.set_ylabel("Train Time (s)", color='tab:green')
    ax3.plot(full_epochs, train_times, label="Train Time", linestyle=':', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    fig.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="upper center", ncol=3)

    plt.title(f"YOLO {model_name} - Loss, mAP & Time")
    fig.tight_layout()
    plt.savefig(f"images/{model_name}/metrics.png")
    plt.close()


model_name = "mamba"
metrics_path = f"metrics/{model_name}/metrics.pth"
assert os.path.exists(metrics_path), "No saved metrics found."

metrics = torch.load(metrics_path)
train_losses = metrics['train_losses']
val_losses = metrics['val_losses']
train_mAP = metrics['train_mAP']
val_mAP = metrics['val_mAP']
train_times = metrics['train_times']

# plot_metrics(train_losses, train_mAP, val_losses, val_mAP, train_times, model_name)
print(train_losses)
print(val_losses)
print(train_mAP)
print(val_mAP)
print(train_times)