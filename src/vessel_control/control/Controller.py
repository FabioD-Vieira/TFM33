import math
from abc import abstractmethod

import numpy as np
from sklearn.neighbors import NearestNeighbors


class Controller:
    def __init__(self, trajectory, AV, AC_max_power):
        self.trajectory = trajectory
        # self.angles = angles

        self.AV = AV
        self.AC_max_power = AC_max_power

        self.knn = NearestNeighbors(n_neighbors=1)
        self.knn.fit(trajectory)

    def execute(self, position, orientation):
        min_error_index = self.knn.kneighbors([position], return_distance=False)[0][0]
        target_point = self.trajectory[min_error_index]

        distance_vector = (target_point[0] - position[0], target_point[1] - position[1])

        s = np.sign(distance_vector[1])
        d = math.sqrt((distance_vector[0] ** 2) + (distance_vector[1] ** 2))
        e = s * d

        AC = self.calculate_AC(e)

        if AC > self.AC_max_power:
            AC = self.AC_max_power
        elif AC < -self.AC_max_power:
            AC = -self.AC_max_power

        # Valor negativo para virar à esquerda quando cruzar a linha
        if orientation > 45:
            return self.AV, -(abs(AC)+1)

        # Valor negativo para virar à direita quando cruzar a linha
        if orientation < -45:
            return self.AV, abs(AC)+1

        return self.AV, AC

    @abstractmethod
    def calculate_AC(self, e):
        pass
