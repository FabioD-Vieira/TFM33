import glob

import cv2
import numpy as np

from src import pool_utils
from src.camera import Camera
from src.system import System

system = System()
camera = Camera()

images = glob.glob('../../images/calibration/*.jpg')
camera.calibrate(images)

img = cv2.imread("img02_3leds.jpg")
undistorted = camera.un_distort(img, balance=1.0)
# cv2.imshow("Undistorted", undistorted)

src_points = np.array([[0, 271], [215, 191], [415, 194], [578, 275]])
system.calculate_homography_matrix(src_points)
reprojected = system.apply_homography(undistorted)

# cv2.imshow("Homography", reprojected)

x, y, angle = pool_utils.get_vessel_info(reprojected)
print(x, y, angle)

x, y, angle = system.get_vessel_info(reprojected)
print(x, y, angle)

# x = np.round(x).astype(int)
# y = np.round(y).astype(int)
# reprojected[y][x] = (255, 255, 255)
cv2.imshow("points", reprojected)

cv2.waitKey(0)
cv2.destroyAllWindows()
