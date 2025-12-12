from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('street.mp4')

id_map = {}
nex_id = 1

trail = defaultdict(lambda: deque(maxlen=50))
apper = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:  # Check if the frame was read successfully
        break

    results = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated_image = frame.copy()  # Directly use the original frame

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.numpy()
        ids = results[0].boxes.id.numpy()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Ensure cx and cy are integers

            apper[oid] += 1
            if apper[oid] >= 5 and oid not in id_map:
                id_map[oid] = nex_id
                nex_id += 1

            if oid in id_map:  # Check if oid exists
                sid = id_map[oid]
                trail[oid].append((cx, cy))
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, f'ID {sid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.circle(annotated_image, (cx, cy), 1, (0, 0, 255), 2)  # Now cx and cy are integers

    cv2.imshow('result', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()