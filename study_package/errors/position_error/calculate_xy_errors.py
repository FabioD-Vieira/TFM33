import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.image_processing.setup.setup import Setup

camera_resolution = (640, 480)
setup = Setup(camera_resolution, balance=0.9)

# Calibrate system for the camera
images = glob.glob('../../../images/calibration/*.jpg')
setup.calibrate_camera([cv2.imread(image_name) for image_name in images])

base_image = cv2.imread("../../../images/corners/POS4_5_4000_01.jpg")
img = cv2.imread("../../../images/corners/img4LUT.jpg")  # This image is read from the camera

setup.calculate_homography_matrix(base_image, img)

normalized_image = np.zeros([480, 640, 3], dtype=np.float32)
for i in range(len(normalized_image)):
    for j in range(len(normalized_image[i])):
        normalized_image[i][j] = (1, i / (480 - 1), j / (640 - 1))
# cv2.imshow("normalized_image", normalized_image)

reprojected = setup.no_lut_process(normalized_image)
# cv2.imshow("normalized_reproject", reprojected)

r = cv2.resize(reprojected, (250, 100))

ex=np.zeros((100,249)).astype('float32')
for i in range(100):
    for j in range(249):
        pi=r[i,j,1]*480
        pj=r[i,j,2]*640
        pi2=r[i,j+1,1]*480
        pj2=r[i,j+1,2]*640
        ex[i,j]= 100 / (np.sqrt((pi2-pi)**2+(pj2-pj)**2) + 0.0000001)

ey=np.zeros((99,250)).astype('float32')
for i in range(99):
    for j in range(250):
        pi=r[i,j,0]*480
        pj=r[i,j,1]*640
        pi2=r[i+1,j,0]*480
        pj2=r[i+1,j,1]*640
        ey[i,j]= 100 / (np.sqrt((pi2-pi)**2+(pj2-pj)**2) + 0.0000001)

ex = cv2.blur(ex, (5, 5))
ey = cv2.blur(ey, (5, 5))

# ey = ey/10
ey = np.ceil(ey/10)*10
ex = np.ceil(ex/10)*10
ey = ey[:, :249]
ex = ex[:99]
# ey = cv2.blur(ey, (3, 3))

result = np.maximum(ex[:99], ey[:, :249])

plt.matshow(result)
plt.matshow(ex)
plt.matshow(ey)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
