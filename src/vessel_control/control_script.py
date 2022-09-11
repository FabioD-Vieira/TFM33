from src.vessel_control.control.PDController import PDController
from src.vessel_control.control.trajectory import create_line
from src.vessel_control.simulations.simulation import Simulation

pool_dim = (25, 10)

# trajectory, angles = create_curve()
trajectory, angles = create_line(pool_dim)


K_orientation = 0.1
KP_position = 0.08
KD_position = 0.08
AV_power = 20
AC_max_power = 20
control = PDController(trajectory, angles, KP_position, KD_position, K_orientation, AV_power, AC_max_power)

simulation = Simulation("", trajectory, pool_dim=pool_dim)

position = (1, 4)
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
