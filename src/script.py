import glob

import cv2

from src.camera import Camera
from src.control.p_control import PControl
from src.control.pd_control import PDControl
from src.shape.circle import Circle
from src.system import System

camera_resolution = (640, 480)
pool_dim = (25, 10)

camera = Camera(camera_resolution, balance=0.9)

circle = Circle(number_of_checkpoints=20, radius=3, center=(pool_dim[0]/2, pool_dim[1]/2))
# control = PDControl(circle, position_threshold=1, orientation_threshold=1)
control = PControl(circle, position_threshold=1, orientation_threshold=1)

system = System(camera, control, pool_dim)

images = glob.glob('../images/calibration/*.jpg')
system.calibrate_camera([cv2.imread(image_name) for image_name in images])

print("Finished setup")

# img = cv2.imread("../images/pool/pool.jpeg")
# img = cv2.imread("../images/pool/pool_leds.jpeg")

base_image = cv2.imread("../images/corners/POS4_5_4000_01.jpg")
img = cv2.imread("../images/corners/POS4_5_4000_02.jpg")
# img = cv2.imread("../images/leds/img01.jpg")

try:
    system.generate_lut(base_image, img)
    print("Generated LUT")
except AssertionError as e:
    print()
    print("Error message: " + str(e))
    raise


img = cv2.imread("../images/leds/img13.jpg")
try:
    system.process(img)
    # system.process2(img)

except AssertionError as e:
    print()
    print("Error message: " + str(e))

print()
print("Finished")
# print(num)

cv2.waitKey(0)
cv2.destroyAllWindows()
