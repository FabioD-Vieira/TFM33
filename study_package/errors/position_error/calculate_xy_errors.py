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

reprojected = setup.no_lut_process(normalized_image)

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
ex = np.ceil(ex/10)*10
ex = ex[:99]
fig1 = plt.figure()
fig1.suptitle("x (mm)", fontsize=20)
ax = fig1.add_subplot(111)
ax.tick_params(labelsize=20)
c1 = ax.matshow(ex)
cbar1 = fig1.colorbar(c1, location="left")
cbar1.set_label(label="mm", size=20)
cbar1.ax.tick_params(labelsize=20)


ey = cv2.blur(ey, (5, 5))
ey = np.ceil(ey/10)*10
ey = ey[:, :249]
fig2 = plt.figure()
fig2.suptitle("y (mm)", fontsize=20)
ax2 = fig2.add_subplot(111)
ax2.tick_params(labelsize=20)
c2 = ax2.matshow(ey)
cbar2 = fig2.colorbar(c2, location="left")
cbar2.set_label(label="mm", size=20)
cbar2.ax.tick_params(labelsize=20)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.tick_params(labelsize=20)
result = np.maximum(ex[:99], ey[:, :249])
c3 = ax3.matshow(result)
cbar3 = fig3.colorbar(c3, location="left")
cbar3.set_label(label="mm", size=20)
cbar3.ax.tick_params(labelsize=20)

plt.show()
