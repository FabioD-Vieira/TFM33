import numpy as np

from src.control.control import Control
from src.homography import Homography
from src.lut import LUT
from src.pool_utils import PoolUtils


class System:

    def __init__(self, camera, control: Control, pool_dim):
        self.__camera = camera
        self.__cam_width, self.__cam_height = self.__camera.dim()

        self.__pool_dim = pool_dim

        self.homography = Homography(self.__camera.dim())
        self.__lut = LUT(self.__camera, self.homography)

        self.__control = control

    def calibrate_camera(self, calibration_images):
        self.__camera.calibrate(calibration_images)

    def generate_lut(self, base_image, img):
        self.__lut.generate_lut(base_image, img)

    def process(self, image):
        x, y, angle = PoolUtils.get_vessel_info(image)
        x, y = self.__lut.apply(int(np.round(y)), int(np.round(x)))

        x *= self.__pool_dim[0]
        y *= self.__pool_dim[1]

        self.__control.get_output(x, y, angle)

    # def process2(self, img):
    #     undistorted = self.__camera.un_distort(img)
    #     rotated = cv2.rotate(undistorted, cv2.ROTATE_180)
    #
    #     cv2.imshow("rotated", rotated)
    #     reprojected = self.homography.apply_homography(rotated)
    #     cv2.imshow("No LUT", reprojected)
