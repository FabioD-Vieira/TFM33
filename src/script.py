import glob

import cv2

from src.system import System

system = System()

images = glob.glob('../images/calibration/*.jpg')
system.calibrate(images)  # Called only once to calibrate

img = cv2.imread("../images/pool/img06_fake_leds.jpg")
# cv2.imshow("Original", img)

undistorted = system.un_distort(img, balance=0.9)
# cv2.imshow("Undistorted", undistorted)


# def print_coordinates(event, x_coord, y_coord, _, _1):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         print(x_coord, y_coord)
#
#
# cv2.setMouseCallback("Undistorted", print_coordinates)

system.calculate_homography_matrix()  # Called only once as well
reprojected = system.apply_homography(undistorted)

cv2.imshow("Homography", reprojected)

x, y, angle = system.get_vessel_info(reprojected)
print(x, y, angle)

cv2.waitKey(0)
cv2.destroyAllWindows()
