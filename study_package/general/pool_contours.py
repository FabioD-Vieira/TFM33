import numpy as np
import cv2

# imread() returns BGR
RED = 2
GREEN = 1
BLUE = 0

image_path = "../../images/pool.jpeg"
image = cv2.imread(image_path)
image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

image_red_channel = image[:, :, RED]
cv2.imshow("red", image_red_channel)

ret, image_binary = cv2.threshold(image_red_channel, 45, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Binary", image_binary)

canny = cv2.Canny(image_binary, 100, 200)
cv2.imshow("Canny", canny)

lines = cv2.HoughLinesP(canny, 1, np.pi/180, 120, minLineLength=10, maxLineGap=250)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)


cv2.imshow("hough", image)


cv2.waitKey(0)
cv2.destroyAllWindows()
