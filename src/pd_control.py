import math
from time import perf_counter_ns

import numpy as np


class PDControl:

    def __init__(self, width, height, pool_dim, position_threshold, orientation_threshold):
        pool_circle_radius = 3
        self.__width, self.__height = width, height
        self.__radius = np.round(pool_circle_radius * self.__width / pool_dim[0]).astype(int)

        self.__position_threshold = position_threshold
        self.__orientation_threshold = orientation_threshold

        self.__center_x = self.__width / 2
        self.__center_y = self.__height / 2

        self.__circle = None
        self.__sorted_points = None
        self.__points_orientations = None

        self.__current_index = 0
        self.__previous_time = perf_counter_ns()

        # Position
        self.__previous_position_error = 0
        self.__position_error = None
        self.__position_KP = 0.1
        self.__position_KD = 0.1

        # Orientation
        self.__previous_orientation_error = 0
        self.__orientation_error = None
        self.__orientation_KP = 0.1
        self.__orientation_KD = 0.1

    def create_circle(self, number_of_checkpoints):
        arc_angles = np.linspace(0 * np.pi, 2 * np.pi, number_of_checkpoints)

        arc_xs = (self.__radius * np.cos(arc_angles)) + self.__center_x

        arc_ys = (self.__radius * np.sin(arc_angles)) + self.__center_y
        arc_ys = arc_ys[::-1]

        self.__circle = np.stack((arc_ys, arc_xs), axis=1)

    def __sort_points(self, vessel_point):
        distances = np.sqrt(np.sum((self.__circle - vessel_point) ** 2, axis=1))
        min_error_index = np.argmin(distances)

        self.__sorted_points = np.roll(self.__circle, -min_error_index, axis=0)

    def __calculate_points_orientations(self):

        theta = np.deg2rad(90)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        vectors_to_center = self.__sorted_points - [self.__center_y, self.__center_x]

        self.__points_orientations = np.zeros(len(self.__sorted_points))
        for i in range(len(vectors_to_center)):
            vector = np.dot(rot, vectors_to_center[i])
            self.__points_orientations[i] = math.degrees(math.atan2(vector[0], vector[1]))

    def __get_position_output(self, vessel_point, time_delta):

        target_position = self.__sorted_points[self.__current_index]
        self.__position_error = target_position - vessel_point

        position_derivative = (self.__position_error - self.__previous_position_error) / time_delta
        self.__previous_position_error = self.__position_error

        return self.__position_KP * self.__position_error + self.__position_KD * position_derivative

    def __get_orientation_output(self, vessel_orientation, time_delta):

        if self.__points_orientations is None:
            self.__calculate_points_orientations()

        point_orientation = self.__points_orientations[self.__current_index]
        self.__orientation_error = point_orientation - vessel_orientation

        orientation_derivative = (self.__orientation_error - self.__previous_orientation_error) / time_delta
        self.__previous_orientation_error = self.__orientation_error

        return self.__orientation_KP * self.__orientation_error + self.__orientation_KD * orientation_derivative

    def get_output(self, x, y, vessel_orientation):
        vessel_point = (y, x)

        if self.__sorted_points is None:
            self.__sort_points(vessel_point)

        time_delta = perf_counter_ns() - self.__previous_time
        self.__previous_time = time_delta

        position_output = self.__get_position_output(vessel_point, time_delta)
        orientation_output = self.__get_orientation_output(vessel_orientation, time_delta)

        position_error_norm = math.sqrt(self.__position_error[0] ** 2 + self.__position_error[1] ** 2)
        if position_error_norm < self.__position_threshold and self.__orientation_error < self.__orientation_threshold:
            self.__current_index += 1

        # to reset behaviour
        if self.__current_index == len(self.__sorted_points):
            self.__current_index = 0

        print(position_output)  # y = height = index 0, x = width = index 1
        print(orientation_output)
