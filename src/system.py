import cv2
import numpy as np

from src import pool_utils
from src.camera import Camera


class System:

    def __init__(self):
        self.__camera = Camera()

        self.__h = None

        # assuming pool size of 25 per 10 meters
        self.__pool_width = 10
        self.__pool_length = 25
        # ratio = self.__pool_width / self.__pool_length

        self.__width = 640
        self.__length = 480  # self.__width * ratio
        self.__initial_point = 0  # top left / margin

        self.__gradient_image = np.zeros([self.__length, self.__width, 3], dtype=np.float32)
        for i in range(len(self.__gradient_image)):
            for j in range(len(self.__gradient_image[i])):
                self.__gradient_image[i][j] = (1, i/(self.__length-1), j/(self.__width-1))

        self.__lut = np.zeros([self.__length, self.__width, 3], dtype=np.uint)

    def setup(self, calibration_images, balance):
        self.__camera.calibrate(calibration_images)
        gradient_undistorted = self.__camera.un_distort(self.__gradient_image, balance)

        self.calculate_homography_matrix()
        gradient_reprojected = self.apply_homography(gradient_undistorted)

        gradient_reprojected[:, :, 1] *= (self.__length - 1)
        gradient_reprojected[:, :, 2] *= (self.__width - 1)

        for x in range(len(gradient_reprojected)):
            for y in range(len(gradient_reprojected[x])):
                self.__lut[x][y] = np.round(gradient_reprojected[x][y]).astype(int)

    def process(self, image):
        new_image = np.array(image, dtype=np.uint8)
        # self.__lut = self.__lut.astype(int)
        for x in range(len(self.__lut)):
            for y in range(len(self.__lut[x])):
                _, row, column = self.__lut[x][y]
                new_image[x][y] = image[row][column]

        return new_image

    def calculate_homography_matrix(self):

        # Pool: 2      3
        #
        #       1      4
        # Obtained by clicking in the image (pool corners)
        src_points = np.array([[86, 78], [263, 129], [435, 264], [566, 469]])

        dst_points = np.array([[self.__initial_point, self.__initial_point + self.__length],
                               [self.__initial_point, self.__initial_point],
                               [self.__initial_point + self.__width - 1, self.__initial_point],
                               [self.__initial_point + self.__width - 1, self.__initial_point + self.__length - 1]])

        self.__h, _ = cv2.findHomography(src_points, dst_points)

    def apply_homography(self, image):
        return cv2.warpPerspective(image, self.__h, (image.shape[1], image.shape[0]))

    def __get_coordinates(self, location_in_image):
        location_image_no_margin = location_in_image - self.__initial_point

        x = location_image_no_margin[1] * self.__pool_length / self.__width
        y = location_image_no_margin[0] * self.__pool_width / self.__length

        return x, y

    def get_vessel_info(self, image):
        back_point, front_point = pool_utils.get_location_in_image(image)

        if back_point is None or front_point is None:
            return None, None, None

        location_in_image = np.round((back_point + front_point) / 2)

        x, y = self.__get_coordinates(location_in_image)
        angle = pool_utils.get_orientation(back_point, front_point)

        return x, y, angle
