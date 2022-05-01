import numpy as np

from src import pool_utils
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
        return self.__lut.apply_lut(image)

    def __get_coordinates(self, location_in_image):
        x = location_in_image[0] * self.__pool_dim[2] / self.__cam_width
        y = location_in_image[1] * self.__pool_dim[1] / self.__cam_height

        return x, y

    def get_vessel_info(self, image):
        x, y, angle = pool_utils.get_vessel_info(image)
        world_x, world_y = self.__get_coordinates(np.array([x, y]))

        return world_x, world_y, angle
