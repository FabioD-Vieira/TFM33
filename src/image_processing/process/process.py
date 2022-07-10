import math

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

    def start(self, image):

        try:
            point_a, point_b, point_c = self.__pool_utils.get_points(image)

            point_a = self.__apply_lut(point_a)
            point_b = self.__apply_lut(point_b)
            point_c = self.__apply_lut(point_c)

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

        # print(self.__last_point)
        # print(self.__last_angle)

        return self.__last_point, self.__last_angle

    def debug(self, img, pA, pB, pC):

        pA = (pA[0] * 640 / 25, pA[1] * 480 / 10)
        img[int(round(pA[1]))][int(round(pA[0]))] = (255, 255, 255)
        img[int(round(pA[1] + 1))][int(round(pA[0]))] = (255, 255, 255)
        img[int(round(pA[1]))][int(round(pA[0] + 1))] = (255, 255, 255)
        img[int(round(pA[1] - 1))][int(round(pA[0]))] = (255, 255, 255)
        img[int(round(pA[1]))][int(round(pA[0] - 1))] = (255, 255, 255)

        pB = (pB[0] * 640 / 25, pB[1] * 480 / 10)
        img[int(round(pB[1]))][int(round(pB[0]))] = (255, 255, 255)
        img[int(round(pB[1] + 1))][int(round(pB[0]))] = (255, 255, 255)
        img[int(round(pB[1]))][int(round(pB[0] + 1))] = (255, 255, 255)
        img[int(round(pB[1] - 1))][int(round(pB[0]))] = (255, 255, 255)
        img[int(round(pB[1]))][int(round(pB[0] - 1))] = (255, 255, 255)

        pC = (pC[0] * 640 / 25, pC[1] * 480 / 10)
        img[int(round(pC[1]))][int(round(pC[0]))] = (255, 255, 255)
        img[int(round(pC[1] + 1))][int(round(pC[0]))] = (255, 255, 255)
        img[int(round(pC[1]))][int(round(pC[0] + 1))] = (255, 255, 255)
        img[int(round(pC[1] - 1))][int(round(pC[0]))] = (255, 255, 255)
        img[int(round(pC[1]))][int(round(pC[0] - 1))] = (255, 255, 255)

    def start_debug(self, image):

        # debug_image = np.zeros(image.shape)
        debug_image2 = np.zeros(image.shape)
        debug_image3 = np.zeros(image.shape)

        try:
            point_a, point_b, point_c = self.__pool_utils.get_points(image)

            # debug_image[int(round(point_a[1]))][int(round(point_a[0]))] = (255, 255, 255)
            # debug_image[int(round(point_b[1]))][int(round(point_b[0]))] = (255, 255, 255)
            # debug_image[int(round(point_c[1]))][int(round(point_c[0]))] = (255, 255, 255)
            # cv2.imshow("points", debug_image)

            point_a = self.__apply_lut(point_a)
            point_b = self.__apply_lut(point_b)
            point_c = self.__apply_lut(point_c)
            self.debug(debug_image3, point_a, point_b, point_c)

            back, front = self.__pool_utils.find_back_and_front(point_a, point_b, point_c)
            # debug_image3[int(round(front[1] * 480 / 10))][int(round(front[0] * 640 / 25))] = (0, 255, 0)
            x, y, angle = self.__pool_utils.get_vessel_info(back, front)

            self.__last_point = (x, y)
            self.__last_angle = angle

        except AssertionError as e:
            print("Error message: " + str(e))
            print("Using last known point and orientation")

        # do something with
        # self.__last_point
        # self.__last_angle

        # print(self.__last_point)
        # print(self.__last_angle)

        if self.__last_point is not None:

            p = (self.__last_point[0] * 640 / 25, self.__last_point[1] * 480 / 10)
            end_x = p[0] + math.cos(math.radians(self.__last_angle)) * 30
            end_y = p[1] + math.sin(math.radians(self.__last_angle)) * 30
            p = (int(round(p[0])), int(round(p[1])))
            p2 = int(round(end_x)), int(round(end_y))
            cv2.arrowedLine(debug_image2, p, p2, (255, 255, 255), 2, tipLength=0.5)
        # cv2.imshow("afterLUT.png", debug_image3)
        cv2.imshow("arrow", debug_image2)

        return self.__last_point, self.__last_angle

    def without_lut(self, image):

        try:
            point_a, point_b, point_c = self.__pool_utils.get_points(image)

            back, front = self.__pool_utils.find_back_and_front(point_a, point_b, point_c)
            x, y, angle = self.__pool_utils.get_vessel_info(back, front)

            self.__last_point = (x * (25/640), y * (10/480))
            self.__last_angle = angle

        except AssertionError as e:
            print("Error message: " + str(e))
            print("Using last known point and orientation")

        # print(self.__last_point)
        # print(self.__last_angle)

    def test_lut(self, point):
        _, y, x = self.__lut[int(np.round(point[1]))][int(np.round(point[0]))]
        return x, y
