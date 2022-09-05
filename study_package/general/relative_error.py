import glob
import math

import cv2
import numpy as np

from src.image_processing.process.process import Process
from src.image_processing.setup.camera import Camera
from src.image_processing.setup.setup import Setup

camera_resolution = (640, 480)
pool_dim = (25, 10)

camera = Camera(camera_resolution, balance=0.9)
setup = Setup(camera, pool_dim)

process = Process(None, pool_dim)

# Calibrate system for the camera
images = glob.glob('../../images/calibration/*.jpg')
setup.calibrate_camera([cv2.imread(image_name) for image_name in images])

print("Finished scripts")
print()

base_image = cv2.imread("../../images/corners/POS4_5_4000_01.jpg")
img = cv2.imread("../../images/corners/img4LUT.jpg")  # This image is read from the camera

setup.calculate_homography_matrix(base_image, img)

cap = cv2.VideoCapture("../../images/videos/vid01.h264")

points = []
angles = []

i = 0


while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("frame", frame)

    if True:

        reprojected = setup.no_lut_process(frame)
        cv2.imshow("re-proj", reprojected)

        point, angle = process.without_lut(reprojected)

        if point is not None:
            points.append(point)
            angles.append(angle)
    # print(point)
    # print(angle)

    cv2.waitKey(10)
    print(i)
    i += 1

print("points=np.array(" + str(points) + ")")
print("angles=np.array(" + str(angles) + ")")

cap.release()
cv2.destroyAllWindows()
