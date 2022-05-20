from src.vessel_control.control.control import Control
from src.vessel_control.shape.shape import Shape


class PControl(Control):

    def __init__(self, shape: Shape, position_threshold, orientation_threshold):
        super().__init__(shape, position_threshold, orientation_threshold)

        self.__initial_orientation_KP = 0.1
        self.__position_KP = 0.01
        self.__orientation_KP = 0.1

    # def _get_initial_orientation_output(self, vessel_point, vessel_orientation):
    #
    #     target_position = self._shape.get_sorted_points(vessel_point)[0]
    #     vector_to_target = target_position - vessel_point
    #
    #     vector_orientation = math.degrees(math.atan2(vector_to_target[0], vector_to_target[1]))
    #     initial_orientation_error = vector_orientation - vessel_orientation
    #
    #     return self.__initial_orientation_KP * initial_orientation_error

    def get_y_output(self, y, target_y):

        y_error = target_y - y
        return self.__position_KP * y_error

    # def _get_position_output(self, vessel_point):
    #
    #     target_position = self._shape.get_sorted_points(vessel_point)[self._current_index]
    #     self._position_error = target_position - vessel_point
    #
    #     return self.__position_KP * self._position_error
    #
    # def _get_orientation_output(self, vessel_orientation):
    #
    #     point_orientation = self._shape.get_points_orientations()[self._current_index]
    #     self._orientation_error = point_orientation - vessel_orientation
    #
    #     return self.__orientation_KP * self._orientation_error
