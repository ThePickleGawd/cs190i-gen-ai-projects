# YoMAMBA

Yo MAMBA so fat, she crushed every benchmark in YOLOv1.

# Setup

```bash
# Install most compatible torch
UV_TORCH_BACKEND=auto uv pip install torch

# Install mambavision
uv pip install mambavision

# Train
uv run train.py

# Test
uv run test_image.py
uv run test_video.py
```