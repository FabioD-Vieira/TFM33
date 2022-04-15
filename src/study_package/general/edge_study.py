import cv2
from matplotlib import pyplot as plt

RED = 2
GREEN = 1
BLUE = 0

image_path = "../../../images/pool.jpeg"
image = cv2.imread(image_path)

image_red_channel = image[:, :, RED]
image_green_channel = image[:, :, GREEN]
image_blue_channel = image[:, :, BLUE]

canny = cv2.Canny(image_red_channel, 100, 200)
# canny2 = cv2.Canny(image_red_channel, 200, 255)
sobel = cv2.Sobel(src=image_red_channel, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=3)

plt.imshow(canny)
plt.show()
# plt.imshow(canny2)
# plt.show()
plt.imshow(sobel)
plt.show()
