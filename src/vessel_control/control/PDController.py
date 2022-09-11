import math

from sklearn.neighbors import NearestNeighbors


class PDController:
    def __init__(self, trajectory, angles, KP_position, KD_position, K_orientation, AV_power, AC_max_power):
        self.__trajectory = trajectory
        self.__angles = angles

        self.__KP_position = KP_position
        self.__KD_position = KD_position
        self.__K_orientation = K_orientation

        self.__AV_power = AV_power
        self.__AC_max_power = AC_max_power

        self.__time_delta = 0.01
        self.__previous_d = 0

        self.__knn = NearestNeighbors(n_neighbors=1)
        self.__knn.fit(trajectory)

    def execute(self, position, orientation):

        min_error_index = self.__knn.kneighbors([position], return_distance=False)[0][0]
        target_point = self.__trajectory[min_error_index]
        target_angle = self.__angles[min_error_index]

        error_vector = (target_point[0] - position[0], target_point[1] - position[1])
        d = math.sqrt((error_vector[0] ** 2) + (error_vector[1] ** 2))

        orientation_diff = abs(target_angle - orientation)

        position_derivative = (d - self.__previous_d) / self.__time_delta
        self.__previous_d = d

        D = d + self.__K_orientation * orientation_diff
        AC = self.__KP_position * D + self.__KD_position * position_derivative
        AV = self.__AV_power

        if error_vector[1] > 0:
            AC = -AC

        return AV, AC
