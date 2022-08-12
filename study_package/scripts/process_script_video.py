import cv2
import numpy as np

from src.image_processing.process.process import Process

pool_dim = (25, 10)
lut = np.load("lut.npy")

process = Process(lut, pool_dim)

cap = cv2.VideoCapture("../../images/videos/vid07.h264")
# ret, frame = cap.read()
# process.start_debug(frame)
# cv2.waitKey(0)


i = 1


# def print_coordinates(event, x, y, flag, params):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         print("yoo")


while True:
    ret, frame = cap.read()

    if not ret:
        break

    rotated = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imshow("rotated", rotated)
    # if i == 1666 or i == 1741 or i == 1810:
    #     cv2.imwrite("rotated" + str(i) + ".png", rotated)
    # cv2.setMouseCallback("rotated", print_coordinates)
    process.start_debug(frame, i)

    # print(i)
    # i += 1

    cv2.waitKey(5)

cap.release()

cv2.destroyAllWindows()
