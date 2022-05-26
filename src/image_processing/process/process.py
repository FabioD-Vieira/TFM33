import cv2
import numpy as np

from src.image_processing.process.pool_utils import PoolUtils


class Process:

    def __init__(self, lut, pool_dim):
        self.__lut = lut
        self.__pool_dim = pool_dim

        self.__pool_utils = PoolUtils()

        self.__last_point = None
        self.__last_angle = None

    def __apply_lut(self, point):
        _, y, x = self.__lut[int(np.round(point[1]))][int(np.round(point[0]))]
        return x*self.__pool_dim[0], y*self.__pool_dim[1]

    def start(self):

        for i in range(30):
            # read from camera
            image = cv2.imread("../../images/leds/img13_leds.jpg")

            try:
                point_a, point_b, point_c = self.__pool_utils.get_points(image)

                point_a = self.__apply_lut(point_a)
                point_b = self.__apply_lut(point_b)
                point_c = self.__apply_lut(point_c)

                # -90º
                # point_a = np.array([10, 10])
                # point_b = np.array([30, 10])
                # point_c = np.array([20, 50])

                back, front = self.__pool_utils.find_back_and_front(point_a, point_b, point_c)
                x, y, angle = self.__pool_utils.get_vessel_info(back, front)

                self.__last_point = (x, y)
                self.__last_angle = angle

            except AssertionError as e:
                print("Error message: " + str(e))
                print("Using last known point and orientation")

            # do something with
            # self.__last_point
            # self.__last_angle

            break
