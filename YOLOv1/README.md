# YOLOv1

Differences:

1. Each bounding box hax class predictions. (N, S, S, B \* (5+C))
2. Use log scale w/h
3. IOU across each box and find max to match

Goal: DeiT for YOLOv1 using my model as a teacher.

DeiT on PASCAL VOC: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11504.pdf

# YOLOv1ResNet Notes

1. Used pretrained ResNet50 as backbone. Detector as normal without dropout
2. Trained for 150 epochs with frozen backbone @ LR=1e-4
3. Trained for 30 epochs with last backbone layer unfrozen and LAMBDA_CLS=2 @ LR=1e-4
4. Trained for 20 epochs with last backbone layer unfrozen and LAMBDA_CLS=50 @ LR=1e-2
5. Trained for 25 epochs last last two backbone layers unfrozen. LAMBDA_CLS=50 @ LR=1e-2

# Next Steps

1. Use ResNet34 frozen backbone with dropout @ LR=1e-2 w/ warmup
2.
