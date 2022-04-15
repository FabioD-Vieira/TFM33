import numpy as np
import cv2

# imread() returns BGR
RED = 2
GREEN = 1
BLUE = 0

image_path = "../../../images/pool.jpeg"
image = cv2.imread(image_path)
image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

cv2.imshow("original", image)


def print_coordinates(event, x, y, flag, params):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x, y)


src_points = np.array([[9, 241], [543, 214], [210, 44], [416, 44]])
dst_points = np.array([[60, 381], [540, 381], [60, 70], [540, 70]])

cv2.setMouseCallback("original", print_coordinates)
h, status = cv2.findHomography(src_points, dst_points)
im_out = cv2.warpPerspective(image, h, (image.shape[1], image.shape[0]))

cv2.imshow("Homography", im_out)

image_red_channel = im_out[:, :, RED]
cv2.imshow("red", image_red_channel)

ret, image_binary = cv2.threshold(image_red_channel, 45, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Binary", image_binary)

canny = cv2.Canny(image_binary, 100, 200)
cv2.imshow("Canny", canny)

lines = cv2.HoughLinesP(canny, 1, np.pi/180, 130, minLineLength=10, maxLineGap=250)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(im_out, (x1, y1), (x2, y2), (255, 0, 0), 3)


cv2.imshow("Hough", im_out)

cv2.waitKey(0)
cv2.destroyAllWindows()
