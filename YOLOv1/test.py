import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils import batch_to_mAP_list

from model import YOLOv1
from data import VOCDataset
import config
from tqdm import tqdm

## Dataset
test_ds = VOCDataset("val")

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), torch.stack(targets)

test_dataloader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

## Model and Metric
model = YOLOv1().to(config.device)
state_dict = torch.load("checkpoints/model.pth", map_location=config.device)
model.load_state_dict(state_dict)
model.eval()

metric = MeanAveragePrecision()

with torch.no_grad():
    test_loss = 0
    for images, targets in tqdm(test_dataloader, desc='Test', leave=False):
        images, targets = images.to(config.device), targets.to(config.device)

        preds = model(images)

        # Calculate mAP
        preds_list, targets_list = batch_to_mAP_list(preds, targets)
        metric.update(preds=preds_list, target=targets_list)
    print(metric.compute())


# TODO: Save metric