import numpy as np

from src.vessel_control.control.p_control import PControl

pool_dim = (25, 10)

# circle = Circle(number_of_checkpoints=20, radius=3, center=(pool_dim[0]/2, pool_dim[1]/2))
# vessel_control = PDControl(circle, position_threshold=1, orientation_threshold=1)
control = PControl(None, position_threshold=1, orientation_threshold=1)

number_of_checkpoints = 1000

line_x = np.linspace(5, 20, number_of_checkpoints)
print(line_x)

line_y = 5
# line_x = np.array([5, 8, 11, 14, 17, 20])

# control.start()

pool_limit_distance = 2

AV = 0
AC = 0

AV_speed = 50
AC_max_speed = 50

while True:

    target_y = line_y

    # Must be read from somewhere
    x = 10
    y = 7
    angle = 0

    if pool_limit_distance < x < pool_dim[0] - pool_limit_distance and \
            pool_limit_distance < y < pool_dim[1] - pool_limit_distance:
        AV = AV_speed
        AC = control.get_y_output(y, target_y)

    else:
        AV = 0
        AC = 0

    if AC > AC_max_speed:
        AC = AC_max_speed

    elif AC < -AC_max_speed:
        AC = -AC_max_speed

    if angle > 90 or angle < -90:
        AC = -AC

    # TODO: increase AC

    left_engine = AV + AC
    right_engine = AV - AC

    print(left_engine, right_engine)

    break
