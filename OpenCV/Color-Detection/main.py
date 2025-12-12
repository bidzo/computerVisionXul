import cv2
from util import get_limits
from PIL import Image

yellow = [0, 255, 255] # yellow in BGR color space
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # get the values or inteval of the color we want
    lowerLimit, upperLimit = get_limits(color=yellow)
    mask = cv2.inRange(hsvImage,lowerLimit, upperLimit)

    # detect and draw bounding boxes using pillow lib
    mask_ = Image.fromarray(mask) # converting image from nparray to pillow

    bbox = mask_.getbbox() # why we used pilow

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (y1, y2), (x2, y2), (255, 0, 0), 5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()