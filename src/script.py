import glob

import cv2

from src.system import System

system = System(camera_resolution=(640, 480), camera_balance=0.9)

images = glob.glob('../images/calibration/*.jpg')
system.calibrate_camera(images)

print("Finished setup")

img = cv2.imread("../images/pool/pool.jpeg")
system.generate_lut(img)

print("Generated LUT")

for i in range(10):
    img2 = cv2.imread("../images/pool/pool.jpeg")
    new_image = system.process(img2)
    break
    # cv2.imshow("new image", new_image)

print("Finished")

cv2.waitKey(0)
cv2.destroyAllWindows()
