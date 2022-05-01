import cv2
import numpy as np

from src import pool_utils
from src.camera import Camera
from src.homography import Homography


class System:

    def __init__(self, camera_resolution, camera_balance):
        self.__camera = Camera(camera_resolution, camera_balance)

        self.__cam_width, self.__cam_height = self.__camera.dim()

        # pool: 25 x 10 meters
        self.__pool_dim = (25, 10)
        self.__homography = Homography(self.__cam_width, self.__cam_height, self.__pool_dim)

        self.__lut = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.uint)

    def __generate_lut(self):

        gradient_image = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.float32)
        for i in range(len(gradient_image)):
            for j in range(len(gradient_image[i])):
                gradient_image[i][j] = (1, i / (self.__cam_height - 1), j / (self.__cam_width - 1))

        gradient_undistorted = self.__camera.un_distort(gradient_image)
        gradient_undistorted = cv2.rotate(gradient_undistorted, cv2.ROTATE_180)

        gradient_reprojected = self.__homography.apply_homography(gradient_undistorted)
        gradient_reprojected[:, :, 1] *= (self.__cam_height - 1)
        gradient_reprojected[:, :, 2] *= (self.__cam_width - 1)

        gradient_reprojected = np.round(gradient_reprojected).astype(int)
        for x in range(len(gradient_reprojected)):
            for y in range(len(gradient_reprojected[x])):
                self.__lut[x][y] = gradient_reprojected[x][y]

    def calibrate_camera(self, calibration_images):
        self.__camera.calibrate(calibration_images)

    def generate_lut(self, img):
        undistorted = self.__camera.un_distort(img)
        rotated = cv2.rotate(undistorted, cv2.ROTATE_180)

        self.__homography.calculate_homography_matrix(rotated)
        self.__generate_lut()

    def process(self, image):
        new_image = np.array(image, dtype=np.uint8)

        for x in range(len(self.__lut)):
            for y in range(len(self.__lut[x])):
                _, row, column = self.__lut[x][y]
                new_image[x][y] = image[row][column]

        return new_image

    def __get_coordinates(self, location_in_image):
        x = location_in_image[0] * self.__pool_dim[2] / self.__cam_width
        y = location_in_image[1] * self.__pool_dim[1] / self.__cam_height

        return x, y

    def get_vessel_info(self, image):
        x, y, angle = pool_utils.get_vessel_info(image)
        world_x, world_y = self.__get_coordinates(np.array([x, y]))

        return world_x, world_y, angle
