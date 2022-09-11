import math

import cv2
import numpy as np

RED = 2
GREEN = 1
BLUE = 0


class PoolUtils:

    def __init__(self):
        self.__number_of_samples = 10
        self.__last_back_points = np.zeros((self.__number_of_samples, 2))
        self.__last_front_points = np.zeros((self.__number_of_samples, 2))

        self.__last_back_point = None
        self.__last_front_point = None

        self.__current_size = 0

        self.__position_outlier_threshold = 1

    @staticmethod
    def get_points(image):

        # Find RED color in image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv_image, (0, 80, 20), (20, 255, 255))
        mask2 = cv2.inRange(hsv_image, (175, 50, 20), (180, 255, 255))
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Find LIGHT in image
        image_channel = image[:, :, RED]
        _, light_mask = cv2.threshold(image_channel, 100, 255, cv2.THRESH_BINARY)

        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # _, gray_mask = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)
        # cv2.imshow("gray_mask", gray_mask)

        # Join both masks and find LED contours
        mask = cv2.bitwise_and(red_mask, light_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.float32))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        assert len(contours) == 3, "No vessel detected"

        point_a, point_b, point_c = [sum(contour) / len(contour) for contour in contours]
        return point_a[0], point_b[0], point_c[0]

    @staticmethod
    def find_back_and_front(point_a, point_b, point_c):

        # Distance between each point
        distance_a_b = math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
        distance_a_c = math.sqrt((point_a[0] - point_c[0]) ** 2 + (point_a[1] - point_c[1]) ** 2)
        distance_c_b = math.sqrt((point_c[0] - point_b[0]) ** 2 + (point_c[1] - point_b[1]) ** 2)

        # Two closest points belong to the front
        if distance_a_b < distance_a_c and distance_a_b < distance_c_b:
            front = np.array([point_a, point_b])
            back = point_c
        elif distance_a_c < distance_a_b and distance_a_c < distance_c_b:
            front = np.array([point_a, point_c])
            back = point_b
        else:
            front = np.array([point_b, point_c])
            back = point_a

        front = sum(front) / 2
        return back, front

    def __is_outlier(self, last_point, point):

        # diff = last_point - point
        diff = (last_point[0] - point[0], last_point[1] - point[1])
        distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)

        return abs(distance) > self.__position_outlier_threshold

    @staticmethod
    def __get_coord_and_angle(back_point, front_point):

        x, y = (front_point + back_point) / 2
        angle = math.degrees(math.atan2(front_point[1] - back_point[1], front_point[0] - back_point[0]))

        return x, y, angle

    def get_vessel_info(self, back_point, front_point):

        outlier = False
        if self.__last_back_point is not None:

            if self.__is_outlier(self.__last_back_point, back_point) and \
                    self.__is_outlier(self.__last_front_point, front_point):
                self.__last_back_point = back_point
                self.__last_front_point = front_point

            if self.__is_outlier(self.__last_back_point, back_point):
                print("back outlier")
                back_point = self.__last_back_point
                outlier = True

            if self.__is_outlier(self.__last_front_point, front_point):
                print("front outlier")
                front_point = self.__last_front_point
                outlier = True

        if not outlier:
            if self.__current_size < self.__number_of_samples:
                self.__last_back_points[self.__current_size] = back_point
                self.__last_front_points[self.__current_size] = front_point
                self.__current_size += 1

            else:
                self.__last_back_points = np.roll(self.__last_back_points, -1, axis=0)
                self.__last_back_points[-1] = back_point

                self.__last_front_points = np.roll(self.__last_front_points, -1, axis=0)
                self.__last_front_points[-1] = front_point

            self.__last_back_point = np.sum(self.__last_back_points, axis=0) / self.__current_size
            self.__last_front_point = np.sum(self.__last_front_points, axis=0) / self.__current_size

        return PoolUtils.__get_coord_and_angle(self.__last_back_point, self.__last_front_point)
