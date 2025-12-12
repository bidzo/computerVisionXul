import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()
    results = model(frame, classes=[0])
    annotated_image = results[0].plot()
    cv2.imshow('result', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()