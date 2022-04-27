import glob

import cv2

from src.system import System

system = System()

images = glob.glob('../images/calibration/*.jpg')
system.setup(images, balance=0.9)

print("Finished setup")

img = cv2.imread("../images/pool/img06_fake_leds.jpg")
new_image = system.process(img)

cv2.imshow("new_image", new_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
