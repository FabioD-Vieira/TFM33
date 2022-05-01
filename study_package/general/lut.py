import glob

import cv2
import numpy as np

from src.camera import Camera
from src.system import System

system = System(camera_resolution=(640, 480), camera_balance=0.9)
camera = Camera(res=(640, 480), balance=0.9)

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

gradient = np.zeros([480, 640, 3], dtype=np.float32)
for i in range(len(gradient)):
    for j in range(len(gradient[i])):
        gradient[i][j] = (1, i/479, j/639)

gradient_undistorted = camera.un_distort(gradient)

gradient_reprojected = cv2.warpPerspective(gradient_undistorted, h, (gradient_undistorted.shape[1], gradient_undistorted.shape[0]))
gradient_reprojected[:, :, 1] *= 479
gradient_reprojected[:, :, 2] *= 639
gradient_reprojected = np.round(gradient_reprojected).astype(int)

lut = np.zeros([480, 640, 3], dtype=np.float32)
for x in range(len(gradient_reprojected)):
    for y in range(len(gradient_reprojected[x])):
        pix = gradient_reprojected[x][y].copy()
        lut[pix[1]][[pix[2]]] = gradient[x][y]

kernel = np.ones((3, 3), np.float32)
lut = cv2.morphologyEx(lut, cv2.MORPH_CLOSE, kernel)

cv2.imshow("lut", lut)

# print(lut[240][320])
print(lut[0][0])
print(img[0][0])

# new_lut = np.array((lut*255), dtype=np.uint8)
# result = cv2.addWeighted(img, 0.7, new_lut, 0.3, 0.0)
# cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
