import glob

import cv2
import numpy as np

from src.camera import Camera
from src.system import System

system = System()
camera = Camera()

images = glob.glob('../images/calibration/*.jpg')
camera.calibrate(images)  # Called only once to calibrate

img = cv2.imread("../images/pool/img06_fake_leds.jpg")
# cv2.imshow("Original", img)

undistorted = camera.un_distort(img, balance=0.9)
# cv2.imshow("Undistorted", undistorted)


# def print_coordinates(event, x_coord, y_coord, _, _1):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         print(x_coord, y_coord)
#
#
# cv2.setMouseCallback("Original", print_coordinates)

system.calculate_homography_matrix()  # Called only once as well
reprojected = system.apply_homography(undistorted)

cv2.imshow("Homography", reprojected)

x, y, angle = system.get_vessel_info(reprojected)
print(x, y, angle)

gradient = np.zeros([480, 640, 3], dtype=np.float32)
for i in range(len(gradient)):
    for j in range(len(gradient[i])):
        gradient[i][j] = (1, i/479, j/639)
# cv2.imshow("test", test)

gradient_undistorted = camera.un_distort(gradient, 0.9)
# cv2.imshow("test2", test2)

gradient_reprojected = system.apply_homography(gradient_undistorted)
gradient_reprojected[:, :, 1] *= 479
gradient_reprojected[:, :, 2] *= 639
gradient_reprojected = np.round(gradient_reprojected).astype(int)

lut = np.zeros([480, 640, 3], dtype=np.float32)
lut2 = np.zeros([480, 640, 3], dtype=np.float32)
for x in range(len(gradient_reprojected)):
    for y in range(len(gradient_reprojected[x])):
        pix = gradient_reprojected[x][y]

        lut[pix[1]][[pix[2]]] = gradient[x][y]
        lut2[x][y] = pix

kernel = np.ones((3, 3), np.float32)
lut = cv2.morphologyEx(lut, cv2.MORPH_CLOSE, kernel)

cv2.imshow("lut", lut)

lut2 = lut2.astype(int)
new_img2 = np.zeros([480, 640, 3], dtype=np.uint8)
for x in range(len(lut2)):
    for y in range(len(lut2[x])):
        _, i, j = lut2[x][y]
        new_img2[x][y] = img[i][j]

cv2.imshow("lut2 new image", new_img2)

# new_lut = np.array((lut*255), dtype=np.uint8)
# result = cv2.addWeighted(img, 0.7, new_lut, 0.3, 0.0)
# cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
