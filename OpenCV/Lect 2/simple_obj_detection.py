import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image = cv2.imread('1.jpg')
image_resized = cv2.resize(image, (640, 480))
results = model(image_resized)

annotated_image = results[0].plot()
cv2.imshow('result', annotated_image)
cv2.waitKey(0)
