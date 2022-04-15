import cv2
import numpy as np

from src import pool_utils


class System:

    def __init__(self):
        self.__h = None

        # assuming pool size of 25 per 10 meters
        self.__pool_width = 10
        self.__pool_length = 25
        ratio = self.__pool_width / self.__pool_length

        self.__length = 450  # pixels for pool length
        self.__width = self.__length * ratio
        self.__initial_point = 80  # top left / margin

    def calculate_homography_matrix(self):

        # Pool: 2      3
        #
        #       1      4
        # Obtained by clicking in the image (pool corners)
        src_points = np.array([[86, 78], [263, 129], [435, 264], [566, 469]])

        dst_points = np.array([[self.__initial_point, self.__initial_point + self.__width],
                               [self.__initial_point, self.__initial_point],
                               [self.__initial_point + self.__length, self.__initial_point],
                               [self.__initial_point + self.__length, self.__initial_point + self.__width]])

        self.__h, _ = cv2.findHomography(src_points, dst_points)

    def apply_homography(self, image):
        return cv2.warpPerspective(image, self.__h, (image.shape[1], image.shape[0]))

    def __get_coordinates(self, location_in_image):
        location_image_no_margin = location_in_image - self.__initial_point

        x = location_image_no_margin[1] * self.__pool_length / self.__length
        y = location_image_no_margin[0] * self.__pool_width / self.__width

        return x, y

    def get_vessel_info(self, image):
        back_point, front_point = pool_utils.get_location_in_image(image)
        location_in_image = np.round((back_point + front_point) / 2)

        x, y = self.__get_coordinates(location_in_image)
        angle = pool_utils.get_orientation(back_point, front_point)

        return x, y, angle
