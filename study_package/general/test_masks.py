import cv2

from src.image_processing.process.pool_utils import PoolUtils

img = cv2.imread("luminosity.png")

PoolUtils.get_points(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
