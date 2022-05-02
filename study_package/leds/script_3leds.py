import glob

import cv2
import numpy as np

from src import pool
from src.camera import Camera
from src.system import System


system = System(camera_resolution=(640, 480), camera_balance=1.0)
camera = Camera(res=(640, 480), balance=1.0)

images = glob.glob('../../images/calibration/*.jpg')
camera.calibrate(images)

src_points = np.array([[86, 78], [263, 129], [435, 264], [566, 469]])
initial_point = 0
width = 640
ratio = 10 / 25
length = np.round(width * ratio).astype(int)
length = 480

dst_points = np.array([[initial_point, initial_point + length - 1],
                       [initial_point, initial_point],
                       [initial_point + width - 1, initial_point],
                       [initial_point + width - 1, initial_point + length - 1]])

h, _ = cv2.findHomography(src_points, dst_points)

img = cv2.imread("../../images/pool/img06_fake_leds.jpg")
undistorted = camera.un_distort(img)
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
