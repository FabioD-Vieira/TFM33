from abc import abstractmethod


class Shape:
    def __init__(self):
        self._points = None
        self._sorted_points = None
        self._points_orientations = None

        self._min_error_index = 0

    def get_points(self):
        return self._points

    def get_sorted_points(self, vessel_points):

        if self._sorted_points is None:
            self.sort_points(vessel_points)

        return self._sorted_points

    def get_points_orientations(self):

        if self._points_orientations is None:
            self.calculate_points_orientations()

        return self._points_orientations

    def get_min_error_index(self):
        return self._min_error_index

    @abstractmethod
    def sort_points(self, vessel_point):
        pass

    @abstractmethod
    def calculate_points_orientations(self):
        pass
