from src.control.control import Control
from src.shape.shape import Shape


class PDControl(Control):

    def __init__(self, shape: Shape, position_threshold, orientation_threshold):
        super().__init__(shape, position_threshold, orientation_threshold)

        self.__previous_position_error = 0
        self.__position_KP = 0.1
        self.__position_KD = 0.1

        self.__previous_orientation_error = 0
        self.__orientation_KP = 0.1
        self.__orientation_KD = 0.1

    def _get_position_output(self, vessel_point, time_delta):

        target_position = self._shape.get_sorted_points(vessel_point)[self._current_index]
        self._position_error = target_position - vessel_point

        position_derivative = (self._position_error - self.__previous_position_error) / time_delta
        self.__previous_position_error = self._position_error

        return self.__position_KP * self._position_error + self.__position_KD * position_derivative

    def _get_orientation_output(self, vessel_orientation, time_delta):

        point_orientation = self._shape.get_points_orientations()[self._current_index]
        self._orientation_error = point_orientation - vessel_orientation

        orientation_derivative = (self._orientation_error - self.__previous_orientation_error) / time_delta
        self.__previous_orientation_error = self._orientation_error

        return self.__orientation_KP * self._orientation_error + self.__orientation_KD * orientation_derivative
