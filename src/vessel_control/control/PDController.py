import math

import numpy as np
from sklearn.neighbors import NearestNeighbors


class PDController:
    def __init__(self, trajectory, KP, KD, AV, AC_max_power):
        self.__trajectory = trajectory

        self.__KP = KP
        self.__KD = KD

        self.__AV = AV
        self.__AC_max_power = AC_max_power

        self.__time_delta = 0.01
        self.__previous_e = 0

        self.__knn = NearestNeighbors(n_neighbors=1)
        self.__knn.fit(trajectory)

    def execute(self, position, orientation):
        min_error_index = self.__knn.kneighbors([position], return_distance=False)[0][0]
        target_point = self.__trajectory[min_error_index]

        distance_vector = (target_point[0] - position[0], target_point[1] - position[1])

        s = np.sign(distance_vector[1])
        d = math.sqrt((distance_vector[0] ** 2) + (distance_vector[1] ** 2))
        e = s * d

        e_derivative = (e - self.__previous_e) / self.__time_delta
        self.__previous_e = e

        AC = self.__KP * e + self.__KD * e_derivative

        if AC > self.__AC_max_power:
            AC = self.__AC_max_power
        elif AC < -self.__AC_max_power:
            AC = -self.__AC_max_power

        # Valor negativo para virar à esquerda quando cruzar a linha
        if orientation > 90:
            return self.__AV, -1

        # Valor negativo para virar à direita quando cruzar a linha
        if orientation < -90:
            return self.__AV, 1

        return self.__AV, AC
