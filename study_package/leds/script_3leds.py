import glob

import cv2
import numpy as np

from src.camera import Camera
from src.system import System
from study_package.orientation import get_orientation

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

cv2.imshow("Homography", reprojected)

hsv_image = cv2.cvtColor(reprojected, cv2.COLOR_BGR2HSV)


# RED
mask = cv2.inRange(hsv_image, (0, 50, 20), (5, 255, 255))
# cv2.imshow("mask", mask)
mask2 = cv2.inRange(hsv_image, (170, 50, 20), (180, 255, 255))
# cv2.imshow("mask2", mask2)

red_mask = cv2.bitwise_or(mask, mask2)
# cv2.imshow("red_mask", red_mask)

# Light
image_channel = reprojected[:, :, 2]

_, light_mask = cv2.threshold(image_channel, 80, 255, cv2.THRESH_BINARY)
# cv2.imshow("light_mask", light_mask)

final_red_mask = cv2.bitwise_and(red_mask, light_mask)
# cv2.imshow("final_red_mask", final_red_mask)

contours, hierarchy = cv2.findContours(final_red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
pointA, pointB, pointC = [sum(contour) / len(contour) for contour in contours]

x, y, angle = get_orientation(pointA[0], pointB[0], pointC[0])
print(x, y, angle)

# back = np.round(back).astype(int)
# front = np.round(front).astype(int)
#
# reprojected[back[1]][back[0]] = (255, 255, 255)
# reprojected[front[1]][front[0]] = (255, 255, 255)
#
# cv2.imshow("points", reprojected)

cv2.waitKey(0)
cv2.destroyAllWindows()
