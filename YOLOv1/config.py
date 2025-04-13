import torch

# Train settings
EPOCHS = 95
DATA_PATH = "data"
BATCH_SIZE = 2
LEARNING_RATE = 1e-4

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

print(f"Using device: {device}")

# Architecture settings based on paper
# https://arxiv.org/pdf/1506.02640

IMG_SIZE = (448, 448)

S = 7
B = 2
C = 20

EPSILON = 1e-6

# 2012
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]