import math
import numpy as np

from src import pool_utils


def get_orientation(point_a, point_b, point_c):

    distance_a_b = math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
    distance_a_c = math.sqrt((point_a[0] - point_c[0]) ** 2 + (point_a[1] - point_c[1]) ** 2)
    distance_c_b = math.sqrt((point_c[0] - point_b[0]) ** 2 + (point_c[1] - point_b[1]) ** 2)

    if distance_a_b < distance_a_c and distance_a_b < distance_c_b:
        front = point_a, point_b
        back = point_c
    elif distance_a_c < distance_a_b and distance_a_c < distance_c_b:
        front = point_a, point_c
        back = point_b
    else:
        front = point_b, point_c
        back = point_a

    front = sum(front) / 2

    x, y = sum([point_a, point_b, point_c]) / 3
    angle = pool_utils.get_orientation(back, front)

    return x, y, angle

# a = np.array([160, 230])
# b = np.array([318, 306])
# c = np.array([394, 134])  # +45 +-
# a = np.array([78, 175])
# b = np.array([171, 114])
# c = np.array([391, 328])  # -45 +-
# a = np.array([150, 313])
# b = np.array([356, 314])
# c = np.array([254, 62])  # -90 +-
# a = np.array([2, 1])
# b = np.array([4, 1])
# c = np.array([3, 5])  # 90
# angle = get_orientation(a, b, c)
# print(angle)
