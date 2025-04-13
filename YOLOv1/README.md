# YOLOv1

Differences:

1. Each bounding box hax class predictions. (N, S, S, B \* (5+C))
2. Use log scale w/h
3. IOU across each box and find max to match

Goal: DeiT for YOLOv1 using my model as a teacher.

DeiT on PASCAL VOC: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11504.pdf
