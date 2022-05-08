import math

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.shape.shape import Shape


class Circle(Shape):

    def __init__(self, number_of_checkpoints, radius, center):
        super().__init__()

        self.__center = center
        self._points = self.__create_circle(number_of_checkpoints, radius)

        self._min_error_index = 0
        self._sorted_points = None
        self._points_orientations = None

    def __create_circle(self, number_of_checkpoints, radius):
        center_x, center_y = self.__center

        # create X points from 0 to 2Pi
        arc_angles = np.linspace(0 * np.pi, 2 * np.pi, number_of_checkpoints)

        # Add correct positions to each point
        arc_xs = (radius * np.cos(arc_angles)) + center_x
        arc_ys = (radius * np.sin(arc_angles)) + center_y

        # Order counter clock wise
        arc_ys = arc_ys[::-1]

        return np.stack((arc_ys, arc_xs), axis=1)

    def sort_points(self, vessel_point):

        # Find the closest point index
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(self._points)

        self._min_error_index = knn.kneighbors([vessel_point], return_distance=False)[0][0]

        # Roll the array in order to have the closest point has the first element
        self._sorted_points = np.roll(self._points, -self._min_error_index, axis=0)

    def calculate_points_orientations(self):

        # Matrix to rotate 90 degrees
        theta = np.deg2rad(90)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Calculate vectors to the center of the circle
        vectors_to_center = self._points - [self.__center[1], self.__center[0]]

        # Calculate tangent vector of each circle point by rotating 90º each vector to the center
        self._points_orientations = np.zeros(len(self._points))
        for i in range(len(vectors_to_center)):
            vector = np.dot(rot, vectors_to_center[i])
            self._points_orientations[i] = math.degrees(math.atan2(vector[0], vector[1]))

        # Roll the array to be consistent with sorted points
        self._points_orientations = np.roll(self._points_orientations, -self._min_error_index, axis=0)
