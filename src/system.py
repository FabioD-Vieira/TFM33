import cv2
import numpy as np

from src import pool
from src.camera import Camera
from src.homography import Homography
from src.lut import LUT
from src.pd_control import PDControl


class System:

    def __init__(self, camera_resolution, camera_balance, pool_dim=(25, 10)):
        self.__cam_width, self.__cam_height = camera_resolution
        self.__camera = Camera(camera_resolution, camera_balance)

        self.__pool_dim = pool_dim

        ratio = self.__pool_dim[1] / self.__pool_dim[0]
        length = np.round(self.__cam_width * ratio).astype(int)
        # length = self.__cam_height

        homography = Homography(self.__cam_width, length)
        self.__lut = LUT(self.__camera, homography)

        self.__control = PDControl(self.__cam_width, length, pool_dim, position_threshold=1, orientation_threshold=1)
        self.__control.create_circle(20)

    def calibrate_camera(self, calibration_images):
        self.__camera.calibrate(calibration_images)

    def generate_lut(self, img):
        self.__lut.generate_lut(img)

    def process(self, image):
        reprojected = self.__lut.apply_lut(image)
        x, y, angle = pool.get_vessel_info(reprojected)
        # x, y = self.__get_coordinates(x, y)

        self.__control.get_output(x, y, angle)

        reprojected[np.round(y).astype(int)][np.round(x).astype(int)] = (0, 0, 255)

        cv2.imshow("reprojected", reprojected)

    def __get_coordinates(self, x, y):
        x *= (self.__pool_dim[0] / self.__cam_width)
        y *= (self.__pool_dim[1] / self.__cam_height)

        return x, y
