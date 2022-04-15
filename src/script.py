import glob

import cv2

from src.pool_processing import PoolProcessing

pool_processing = PoolProcessing()

images = glob.glob('../images/calibration/*.jpg')
pool_processing.calibrate(images)  # Called only once to calibrate

img = cv2.imread("../images/pool/img06_fake_leds.jpg")
# cv2.imshow("Original", img)

undistorted = pool_processing.un_distort(img, balance=0.9)
# cv2.imshow("Undistorted", undistorted)


# def print_coordinates(event, x_coord, y_coord, _, _1):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         print(x_coord, y_coord)
#
#
# cv2.setMouseCallback("Undistorted", print_coordinates)

pool_processing.calculate_homography_matrix()  # Called only once as well
reprojected = pool_processing.apply_homography(undistorted)

cv2.imshow("Homography", reprojected)

x, y = pool_processing.get_vessel_info(reprojected)
print(x, y)

cv2.waitKey(0)
cv2.destroyAllWindows()
