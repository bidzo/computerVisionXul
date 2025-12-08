import cv2

img = cv2.imread('1.jpg')
# image processing
resized = cv2.resize(img, (640, 480))
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
edge = cv2.Canny(img, 50, 150)
# show the images
cv2.imshow('resized', resized)
cv2.imshow('grey', grey)
cv2.imshow('blurred', blurred)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()