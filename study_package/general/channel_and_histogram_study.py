import cv2
from matplotlib import pyplot as plt

# imread() returns BGR
RED = 2
GREEN = 1
BLUE = 0

image_path = "../../images/pool.jpeg"
image = cv2.imread(image_path)
image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)

# cv2.imshow("Original", image)
# cv2.waitKey(0)

image_red_channel = image[:, :, RED]
image_green_channel = image[:, :, GREEN]
image_blue_channel = image[:, :, BLUE]

cv2.imshow("Red Channel", image_red_channel)
cv2.imshow("Green Channel", image_green_channel)
cv2.imshow("Blue Channel", image_blue_channel)

plt.hist(image_red_channel.ravel(), 256, [0, 256])
plt.show()
plt.hist(image_green_channel.ravel(), 256, [0, 256])
plt.show()
plt.hist(image_blue_channel.ravel(), 256, [0, 256])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
