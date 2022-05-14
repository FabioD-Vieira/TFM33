import math
from abc import abstractmethod


class Control:

    def __init__(self, shape, position_threshold, orientation_threshold):
        self._shape = shape
        self.__position_threshold = position_threshold
        self.__orientation_threshold = orientation_threshold

        self._position_error = None
        self._previous_position_error = 0

        self._orientation_error = None
        self._previous_orientation_error = 0

        self._current_index = 0

    def __get_output(self, x, y, vessel_orientation):
        # TODO: Must have in mind initial vessel orientation before computing any position correction

        vessel_point = (y, x)

        # self._get_initial_orientation_output(vessel_point, vessel_orientation)

        position_output = self._get_position_output(vessel_point)
        orientation_output = self._get_orientation_output(vessel_orientation)

        position_error_norm = math.sqrt(self._position_error[0] ** 2 + self._position_error[1] ** 2)
        if position_error_norm < self.__position_threshold and self._orientation_error < self.__orientation_threshold:
            self._current_index += 1

        # to reset behaviour
        if self._current_index == len(self._shape.get_points()):
            self._current_index = 0

        print(position_output)  # y = height = index 0, x = width = index 1
        print(orientation_output)

    def start(self):

        # read x, y and orientation
        x = 10
        y = 5
        angle = 0

        self.__get_output(x, y, angle)

    # @abstractmethod
    # def _get_initial_orientation_output(self, vessel_point, vessel_orientation):
    #     pass

    @abstractmethod
    def _get_position_output(self, vessel_point):
        pass

    @abstractmethod
    def _get_orientation_output(self, vessel_orientation):
        pass
