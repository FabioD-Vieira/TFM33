import sys

import numpy as np

from src.vessel_control.control.PDController import PDController
from src.vessel_control.control.trajectory import create_line, create_curve
from src.vessel_control.simulations.simulation import Simulation

pool_dim = (25, 10)
trajectory, angles = create_line(pool_dim)
# trajectory, angles = create_curve()

simulation = Simulation("", trajectory, pool_dim=pool_dim)


def read_pose():
    return position, orientation


min_error = sys.maxsize
min_error_kp = -1
min_error_kd = -1

kp_options = np.linspace(0.01, 1, 100)
kd_options = np.linspace(0.01, 1, 100)

iterations = 300

for kp in kp_options:
    kp = round(kp, 2)

    for kd in kd_options:
        kd = round(kd, 2)

        AV_power = 20
        AC_max_power = 20
        control = PDController(trajectory, angles, kp, kd, AV_power, AC_max_power)

        position = (1, 1)
        orientation = 0

        i = 0
        total_error = 0

        running = True
        while running:
            # read position and orientation
            position, orientation = read_pose()

            AV, AC, d = control.execute(position, orientation)
            total_error += d

            # send data to vessel
            running, position, orientation = simulation.draw(position, orientation, AV, AC)
            # position, orientation = simulation.update_vessel()

            i += 1
            if i == iterations:
                break

        total_error = total_error / iterations
        print(kp, kd, total_error)
        if total_error < min_error:
            min_error = total_error
            min_error_kp = kp
            min_error_kd = kd

print()
print(min_error)
print(min_error_kp)
print(min_error_kd)
