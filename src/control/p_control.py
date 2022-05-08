from src.control.control import Control
from src.shape.shape import Shape


class PControl(Control):

    def __init__(self, shape: Shape, position_threshold, orientation_threshold):
        super().__init__(shape, position_threshold, orientation_threshold)

        self.__position_KP = 0.1
        self.__orientation_KP = 0.1

    def _get_position_output(self, vessel_point, time_delta):

        target_position = self._shape.get_sorted_points(vessel_point)[self._current_index]
        self._position_error = target_position - vessel_point

        return self.__position_KP * self._position_error

    def _get_orientation_output(self, vessel_orientation, time_delta):

        point_orientation = self._shape.get_points_orientations()[self._current_index]
        self._orientation_error = point_orientation - vessel_orientation

        return self.__orientation_KP * self._orientation_error
