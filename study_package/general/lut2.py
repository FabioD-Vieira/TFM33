import glob

import cv2
import numpy as np

from src.image_processing.setup import Camera
from src.image_processing.setup.setup import Setup

system = Setup(camera_resolution=(640, 480), camera_balance=0.9)
camera = Camera(res=(640, 480), balance=0.9)

images = glob.glob('../../images/calibration/*.jpg')
camera.calibrate(images)

src_points = np.array([[86, 78], [263, 129], [435, 264], [566, 469]])
initial_point = 0
width = 640
ratio = 10 / 25
length = np.round(width * ratio).astype(int)
# length = 480

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

lut2 = np.zeros([480, 640, 3], dtype=np.float32)
for x in range(len(gradient_reprojected)):
    for y in range(len(gradient_reprojected[x])):
        pix = gradient_reprojected[x][y]
        lut2[x][y] = pix

lut2 = lut2.astype(int)
new_img2 = np.zeros([480, 640, 3], dtype=np.uint8)
for x in range(len(lut2)):
    for y in range(len(lut2[x])):
        _, i, j = lut2[x][y]
        new_img2[x][y] = img[i][j]

# print(lut2[0][0])
# print(img[44][205])
# print(new_img2[0][0])
cv2.imshow("lut2 new image", new_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
