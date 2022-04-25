import glob

import cv2
import numpy as np

from src.camera import Camera
from src.system import System

system = System()
camera = Camera()

images = glob.glob('../../images/calibration/*.jpg')
camera.calibrate(images)

img = cv2.imread("img01_resized_gr.jpg")
undistorted = camera.un_distort(img, balance=1.0)
cv2.imshow("Undistorted", undistorted)


def print_coordinates(event, x_coord, y_coord, _, _1):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x_coord, y_coord)


cv2.setMouseCallback("Undistorted", print_coordinates)

src_points = np.array([[0, 249], [162, 192], [387, 207], [530, 283]])
system.calculate_homography_matrix(src_points)
reprojected = system.apply_homography(undistorted)

cv2.imshow("Homography", reprojected)


hsv_image = cv2.cvtColor(reprojected, cv2.COLOR_BGR2HSV)


# RED
mask = cv2.inRange(hsv_image, (0, 50, 20), (5, 255, 255))
# cv2.imshow("mask", mask)
mask2 = cv2.inRange(hsv_image, (175, 50, 20), (180, 255, 255))
# cv2.imshow("mask2", mask2)

red_mask = cv2.bitwise_or(mask, mask2)
# cv2.imshow("red_mask", red_mask)


# GREEN
green_mask = cv2.inRange(hsv_image, (20, 150, 20), (30, 255, 100))
# cv2.imshow("green_mask", green_mask)


# Light
image_channel = reprojected[:, :, 2]

_, light_mask = cv2.threshold(image_channel, 20, 255, cv2.THRESH_BINARY)
# cv2.imshow("light_mask", light_mask)


final_red_mask = cv2.bitwise_and(red_mask, light_mask)
cv2.imshow("final_red_mask", final_red_mask)

final_green_mask = cv2.bitwise_and(green_mask, light_mask)
cv2.imshow("final_green_mask", final_green_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
