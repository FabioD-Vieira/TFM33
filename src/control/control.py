import math
from abc import abstractmethod
from time import perf_counter_ns


class Control:

    def __init__(self, shape, position_threshold, orientation_threshold):
        self._shape = shape
        self.__position_threshold = position_threshold
        self.__orientation_threshold = orientation_threshold

        self.__previous_time = perf_counter_ns()

        self._position_error = None
        self._previous_position_error = 0

        self._orientation_error = None
        self._previous_orientation_error = 0

        self._current_index = 0

    def get_output(self, x, y, vessel_orientation):
        # TODO: Must have in mind initial vessel orientation before computing any position correction

        vessel_point = (y, x)

        time_delta = perf_counter_ns() - self.__previous_time
        self.__previous_time = time_delta

        position_output = self._get_position_output(vessel_point, time_delta)
        orientation_output = self._get_orientation_output(vessel_orientation, time_delta)

        position_error_norm = math.sqrt(self._position_error[0] ** 2 + self._position_error[1] ** 2)
        if position_error_norm < self.__position_threshold and self._orientation_error < self.__orientation_threshold:
            self._current_index += 1

        # to reset behaviour
        if self._current_index == len(self._shape.get_points()):
            self._current_index = 0

        print(position_output)  # y = height = index 0, x = width = index 1
        print(orientation_output)

    @abstractmethod
    def _get_position_output(self, vessel_point, time_delta):
        pass

    @abstractmethod
    def _get_orientation_output(self, vessel_orientation, time_delta):
        pass
