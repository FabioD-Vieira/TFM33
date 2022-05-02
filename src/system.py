import cv2
import numpy as np

from src import pool
from src.camera import Camera
from src.lut import LUT


class System:

    def __init__(self, camera_resolution, camera_balance, pool_dim=(25, 10)):
        self.__cam_width, self.__cam_height = camera_resolution
        self.__camera = Camera(camera_resolution, camera_balance)

        self.__pool_dim = pool_dim
        self.__lut = LUT(self.__camera, self.__pool_dim)

    def calibrate_camera(self, calibration_images):
        self.__camera.calibrate(calibration_images)

    def generate_lut(self, img):
        self.__lut.generate_lut(img)

    def process(self, image):
        reprojected = self.__lut.apply_lut(image)

        x, y, angle = pool.get_vessel_info(reprojected)
        x, y = self.__get_coordinates(x, y)

        print(x, y, angle)

    def __get_coordinates(self, x, y):
        x *= (self.__pool_dim[0] / self.__cam_width)
        y *= (self.__pool_dim[1] / self.__cam_height)

        return x, y

    def get_vessel_info(self, image):
        x, y, angle = pool.get_vessel_info(image)
        world_x, world_y = self.__get_coordinates(np.array([x, y]))

        return world_x, world_y, angle
