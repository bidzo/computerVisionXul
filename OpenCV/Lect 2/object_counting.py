import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('bottle.mp4')
unique_id = set()

while True:
    ret, frame = cap.read()
    # track bottles class 39 see YOLO docs , persist the object ID frame by frame persist = True
    #  dont log anything verbos = False
    #  Chikuita kunge chi inference on that model
    results = model.track(frame, classes=[39], persist=True, verbose=False)

    annotated_image = results[0].plot()

    # if there are boxes do the following and if there is no boxes do nothing
    if results[0].boxes and results[0].boxes.id is not None:
     # define ID for each box and convert to numpy array
        ids = results[0].boxes.id.numpy()
        for oid in ids:
            unique_id.add(oid)
         # add the text showing total bootle count
        cv2.putText(annotated_image,f'Count {len(unique_id)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         # resize the frames
        resized_image = cv2.resize(annotated_image, (640, 480))
        cv2.imshow('result', resized_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()