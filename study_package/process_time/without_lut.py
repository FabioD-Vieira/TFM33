import glob
import time

import cv2

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
img = cv2.imread("../../images/corners/img4LUT_1920x1440.jpg")  # This image is read from the camera

setup.calculate_homography_matrix(base_image, img)


# image = cv2.imread("../../images/corners/img4LUT.jpg")
#
# total = 0
#
# for _ in range(10):
#     start = time.time()
#
#     for _ in range(1000):
#         reprojected = setup.no_lut_process(image)
#
#     diff = time.time() - start
#     print(diff)
#     total += diff
#
# average = total / 10
# print("Average", average)

cv2.waitKey(0)
cv2.destroyAllWindows()
