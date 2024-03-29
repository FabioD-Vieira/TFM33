import glob

import cv2
import numpy as np

from src.image_processing.setup.setup import Setup

# Initializations
camera_resolution = (640, 480)
setup = Setup(camera_resolution, balance=0.9)

# Calibrate system for the camera
images = glob.glob('../../images/calibration/*.jpg')
setup.calibrate_camera([cv2.imread(image_name) for image_name in images])

print("Finished calibration")
print()

# Generate LUT to speed up process
base_image = cv2.imread("../../images/corners/POS4_5_4000_01.jpg")
img = cv2.imread("../../images/corners/img4LUT.jpg")  # This image is read from the camera

setup.calculate_homography_matrix(base_image, img)

try:
    lut = setup.generate_lut()
    np.save("lut", lut)

    print("LUT generated")
except AssertionError as e:
    print(e)

cv2.waitKey(0)
cv2.destroyAllWindows()
