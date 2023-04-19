from src.vessel_control.control.PController import PController
from src.vessel_control.control.PDController import PDController
from src.vessel_control.control.trajectory import create_line, create_curve
from src.vessel_control.simulations.simulation import Simulation

pool_dim = (25, 10)
trajectory, angles = create_line(pool_dim)
# trajectory, angles = create_curve()

simulation = Simulation("", trajectory, pool_dim=pool_dim)

KP = 1.0
KD = 0.4
AV_power = 20
AC_max_power = 20
# control = PController(trajectory, KP, AV_power, AC_max_power)
control = PDController(trajectory, KP, KD, AV_power, AC_max_power)


position = (1, 3)
orientation = 0


def read_pose():
    return position, orientation


running = True
while running:
    # read position and orientation
    position, orientation = read_pose()

    AV, AC = control.execute(position, orientation)

    # send data to vessel
    running, position, orientation = simulation.draw(position, orientation, AV, AC)
