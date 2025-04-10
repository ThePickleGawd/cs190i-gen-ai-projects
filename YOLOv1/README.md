# YOLOv1

Differences:

1. Each bounding box hax class predictions. (N, S, S, B \* (5+C))
2. Use log scale w/h
3. IOU across each box and find max to match
