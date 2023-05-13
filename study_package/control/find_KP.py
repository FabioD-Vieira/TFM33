import sys

import numpy as np

from src.vessel_control.control.PController import PController
from src.vessel_control.trajectory import create_curve
from src.vessel_control.simulations.simulation import Simulation

pool_dim = (25, 10)
# trajectory, angles = create_line(pool_dim)
trajectory, angles = create_curve()

simulation = Simulation("", trajectory, pool_dim=pool_dim)


def read_pose():
    return position, orientation


min_error = sys.maxsize
min_error_kp = -1

kp_options = np.linspace(0.01, 10, 1000)

iterations = 300

for k in kp_options:
    k = round(k, 2)

    KP = k
    AV_power = 20
    AC_max_power = 20
    control = PController(trajectory, angles, KP, AV_power, AC_max_power)

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
    print(k, total_error)
    if total_error < min_error:
        min_error = total_error
        min_error_kp = k

print()
print(min_error)
print(min_error_kp)
