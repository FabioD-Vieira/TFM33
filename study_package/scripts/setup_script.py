import glob

import cv2
import numpy as np

from src.image_processing.setup.camera import Camera
from src.image_processing.setup.setup import Setup

# Initializations
camera_resolution = (640, 480)
pool_dim = (25, 10)

camera = Camera(camera_resolution, balance=0.9)
setup = Setup(camera, pool_dim)

# Calibrate system for the camera
images = glob.glob('../../images/calibration/*.jpg')
setup.calibrate_camera([cv2.imread(image_name) for image_name in images])

# Generate LUT to speed up process
base_image = cv2.imread("../../images/corners/POS4_5_4000_01.jpg")
img = cv2.imread("../../images/corners/img4LUT.jpg")  # This image is read from the camera

try:
    lut = setup.generate_lut(base_image, img)
    cv2.imshow("lut", lut)

    new_lut = np.array((lut * 255), dtype=np.uint8)
    result = cv2.addWeighted(img, 1, new_lut, 0.5, 0.0)
    cv2.imshow("lut_overlay", result)

except AssertionError as e:
    print(e)

cv2.waitKey(0)
cv2.destroyAllWindows()
