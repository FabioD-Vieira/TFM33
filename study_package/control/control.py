import glob

import cv2
import numpy as np

from src.camera import Camera
from src.system import System

system = System()
camera = Camera()

images = glob.glob('../../images/calibration/*.jpg')
camera.calibrate(images)

img = cv2.imread("../../images/pool/img06.jpg")
undistorted = camera.un_distort(img, balance=.9)
cv2.imshow("Undistorted", undistorted)


system.calculate_homography_matrix()
reprojected = system.apply_homography(undistorted)

cv2.imshow("Homography", reprojected)

radius = np.round(3*640/25).astype(int)

circle_img = cv2.circle(reprojected, (319, 128), radius, (0, 0, 255))
cv2.imshow("img_circle", circle_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


