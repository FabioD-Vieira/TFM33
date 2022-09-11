import cv2
import numpy as np

from src.image_processing.process.process import Process

pool_dim = (25, 10)
lut = np.load("lut.npy")

process = Process(lut, pool_dim)

cap = cv2.VideoCapture("../../images/videos/vid07.h264")

while True:
    ret, frame = cap.read()
    cv2.imshow("f", frame)

    if not ret:
        break

    process.start(frame)
    cv2.waitKey(1)

cap.release()


cv2.destroyAllWindows()
