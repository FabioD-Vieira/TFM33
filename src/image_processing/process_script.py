import cv2
import numpy as np

from src.image_processing.process.process import Process

pool_dim = (25, 10)
lut = np.load("lut.npy")

image = cv2.imread("img13_leds.jpg")
cv2.imshow("original", image)

process = Process(lut, pool_dim)
process.start(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
