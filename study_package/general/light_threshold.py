import cv2

RED = 2
GREEN = 1
BLUE = 0

image_path = "../../../images/III/9.jpeg"
image = cv2.imread(image_path)
image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

cv2.imshow("original", image)

image_channel = image[:, :, BLUE]
cv2.imshow("channel", image_channel)

ret, image_binary = cv2.threshold(image_channel, 245, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary", image_binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
