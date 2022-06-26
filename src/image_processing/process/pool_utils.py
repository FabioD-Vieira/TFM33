import math

import cv2
import numpy as np

RED = 2
GREEN = 1
BLUE = 0


class PoolUtils:

    def __init__(self):
        self.__number_of_samples = 10
        self.__last_positions = np.zeros((self.__number_of_samples, 2))
        self.__last_orientations = np.zeros(self.__number_of_samples)

        self.__current_size = 0

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

        # Join both masks and find LED contours
        mask = cv2.bitwise_and(red_mask, light_mask)
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

    def get_vessel_info(self, back, front):

        x, y = (front + back) / 2

        point = (x, y)
        angle = math.degrees(math.atan2(front[1] - back[1], front[0] - back[0]))

        if self.__current_size < self.__number_of_samples:
            self.__last_positions[self.__current_size] = point
            self.__last_orientations[self.__current_size] = angle
            self.__current_size += 1

        else:
            self.__last_positions = np.roll(self.__last_positions, -1, axis=0)
            self.__last_positions[-1] = point

            self.__last_orientations = np.roll(self.__last_orientations, -1, axis=0)
            self.__last_orientations[-1] = angle

        point = np.sum(self.__last_positions, axis=0) / self.__current_size
        angle = np.sum(self.__last_orientations) / self.__current_size

        return point[0], point[1], angle
