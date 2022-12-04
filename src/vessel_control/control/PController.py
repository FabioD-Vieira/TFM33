import math
from abc import ABC

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.vessel_control.control.Controller import Controller


class PController(Controller):
    def __init__(self, trajectory, KP, AV, AC_max_power):
        super().__init__(trajectory, AV, AC_max_power)

        self.__KP = KP

    def calculate_AC(self, e):
        return self.__KP * e
