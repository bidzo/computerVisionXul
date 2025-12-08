import cv2
# capture live camera feed from the webcam
cap = cv2.VideoCapture(0)
frames = []
gap = 5
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break