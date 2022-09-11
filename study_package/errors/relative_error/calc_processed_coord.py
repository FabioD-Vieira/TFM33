import cv2
import numpy as np

from src.image_processing.process.process import Process

pool_dim = (25, 10)
lut = np.load("lut.npy")

cv2.imshow("lut", lut)

process = Process(lut, pool_dim)

cap = cv2.VideoCapture("../../../images/videos/vid01.h264")

points = []
angles = []

list_of_points = []
list_of_angles = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("f", frame)

    point, angle = process.start(frame)

    if point is not None:
        points.append(point)
        angles.append(angle)
    else:
        list_of_points.append(points)
        list_of_angles.append(angles)
        points = []
        angles = []

    # print(point, angle)
    cv2.waitKey(1)
    # break

cap.release()


cv2.destroyAllWindows()

list_of_points.append(points)
list_of_angles.append(angles)

print("points=np.array(" + str(list_of_points) + ")")
print("angles=np.array(" + str(list_of_angles) + ")")
