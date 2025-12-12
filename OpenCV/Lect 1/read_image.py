import cv2

# Load image
image = cv2.imread('1.jpg')

image = cv2.resize(image, (640, 480))

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to binary using thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)  # -1 to draw all contours

# Show the output
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()