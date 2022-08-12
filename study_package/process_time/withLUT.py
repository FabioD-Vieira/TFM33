import time

import cv2
import numpy as np

from src.image_processing.process.process import Process

pool_dim = (25, 10)
lut = np.load("lut_4K_3840x2160.npy")

cv2.imshow("lut", lut)
cv2.waitKey(0)
cv2.destroyAllWindows()

process = Process(lut, pool_dim)

total = 0

for _ in range(10):
    start = time.time()
    for _ in range(1000):
        process.test_lut((200, 200))
    diff = time.time() - start
    print(diff)
    total += diff

average = total / 10
print("Average", average)
