import cv2
import numpy as np

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID'
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

for i in range(100):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out.write(frame)

out.release()
cv2.destroyAllWindows()
