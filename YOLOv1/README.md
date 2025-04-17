# YOLOv1

Differences:

1. Each bounding box hax class predictions. (N, S, S, B \* (5+C))
2. Use log scale w/h
3. IOU across each box and find max to match

Goal: DeiT for YOLOv1 using my model as a teacher.

DeiT on PASCAL VOC: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11504.pdf

# YOLOv1ResNet Notes

1. Used pretrained ResNet50 as backbone. Detector as normal without dropout
2. Trained for 150 epochs with frozen backbone
3. Trained for \_\_\_ epochs with last backbone layer unfrozen and boosting loss for classification
