# CS 190I: Generative AI and Deep Learning

Projects for UCSB CS 190I Spring '25 

## 1. YoMAMBA
A Mamba-based YOLOv1 architecture. We compare performance with ResNet backbone.

![figure_2](/YoMAMBA/images/figure_2.png)

### Prerequisites
```bash
pip install uv

# Install most compatible torch
UV_TORCH_BACKEND=auto uv pip install torch

# Sync packages
uv sync

# Yo Mamba (YOLO) requirements
uv pip install mambavision

# Mo Bamba (Finetuned LLM) Requirements
git submodule update --init --recursive
pip install git+https://github.com/huggingface/transformers.git
```

### Train
Training takes about 12 hours on RTX 3070 Ti GPU. 
```bash
# Modify train.py to specify which model to train
uv run train.py

# Plot train metrics
uv run plot_loss_mAP.py
```

![figure_1](/YoMAMBA/images/figure_1.png)

### Test
Test model on sample images or video
```bash
# Set flag to enable mamba. Default: ResNet backbone
uv run test_image.py --use-mamba
uv run test_video.py --use-mamba
```

### YOLOv2 - Fails Classification
My initial attempt to recreate YOLOv2. I ran into issues with exploding gradients and poor classification (despite good object detection). I've tried the following:
- Boosting classification loss
- Modifying the loss function to guarentee each target bbox gets assigned a predicted bbox
- Using log scale predictions for width and height
- Using sigmoid on x,y,conf to clamp values

| Case 1 | Case 2 | Case 3 |
|--------|--------|--------|
| ![](YOLOv2/images/paper/frozen_backbone_bad_classification.png) | ![](YOLOv2/images/paper/frozen_backbone_bad_classification_2.png) | ![](YOLOv2/images/paper/frozen_backbone_bad_classification_many.png) |

#### Train Stats. Unfreezed final backbone layer after 150 epochs

![YOLOv2_Loss](/YOLOv2/images/YOLOv1ResNet/metrics_260_epochs.png)

## 2. Soon to come!

## Utils

### Running Batches on CSIL
```bash
# Connect to GPU
srun --gpus=1 --nodes=1 --time=24:00:00 --cpus-per-task=4 --pty bash

# See job queue
squeue | grep dylanlu

# Cancel job ID, find with squeue
scancel [job-id]

# Keep terminal alive
tmus new -s example
tmux ls
tmux a -t example # Attach

# Run batch
sbatch sbatch_script.sh
```
