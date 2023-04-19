from src.vessel_control.control.Controller import Controller


class PDController(Controller):
    def __init__(self, trajectory, KP, KD, AV, AC_max_power):
        super().__init__(trajectory, AV, AC_max_power)

        self.__KP = KP
        self.__KD = KD

        self.__time_delta = 0.01
        self.__previous_e = 0

    def calculate_AC(self, e):
        e_derivative = (e - self.__previous_e) / self.__time_delta
        self.__previous_e = e

        return self.__KP * e + self.__KD * e_derivative
