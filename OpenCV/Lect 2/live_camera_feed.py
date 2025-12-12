import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("street.mp4") # 0 camera capture
model = YOLO('yolov8n.pt') # YOLO v8 Pretrained Model

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated_image = results[0].plot()
    cv2.imshow('Live Camera Feed', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()