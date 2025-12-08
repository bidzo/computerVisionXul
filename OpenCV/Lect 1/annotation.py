import cv2
import numpy as np

canvas = np.zeros((512, 512, 3), dtype="uint8")
cv2.line(canvas, (0, 0), (512, 512), (255, 0, 0), 2)
cv2.rectangle(canvas, (0, 0), (512, 512), (255, 0, 0), 2)
cv2.putText(canvas, 'Text pano !!!!', (10,500), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
cv2.imshow('canvas', canvas)
cv2.waitKey(0)