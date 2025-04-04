# Train settings
EPOCHS = 32
DATA_PATH = "data"
BATCH_SIZE = 64

# Architecture settings based on paper
# https://arxiv.org/pdf/1506.02640

IMG_SIZE = (448, 448)

S = 7
B = 2
C = 20

# 2012
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]