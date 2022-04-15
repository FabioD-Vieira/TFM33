import math
import numpy as np


# Assume um referencial a começar no ponto mais à esquerda, em baixo
# O eixo y de cada ponto deve ser adaptado para o referencial no topo
def get_orientation(point_a, point_b, point_c):

    distance_a_b = math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
    distance_a_c = math.sqrt((point_a[0] - point_c[0]) ** 2 + (point_a[1] - point_c[1]) ** 2)
    distance_c_b = math.sqrt((point_c[0] - point_b[0]) ** 2 + (point_c[1] - point_b[1]) ** 2)

    if distance_a_b < distance_a_c and distance_a_b < distance_c_b:
        back = point_a, point_b
        front = point_c
    elif distance_a_c < distance_a_b and distance_a_c < distance_c_b:
        back = point_a, point_c
        front = point_b
    else:
        back = point_b, point_c
        front = point_a

    middle_point = (back[0] + back[1]) // 2

    print(front)
    print(middle_point)

    # inverted y-axis
    # front[1], middle_point[1] = middle_point[1], front[1]

    angle_radians = math.atan2(front[1] - middle_point[1], front[0] - middle_point[0])
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


# a = np.array([160, 230])
# b = np.array([318, 306])
# c = np.array([394, 134])  # +45 +-
# a = np.array([78, 175])
# b = np.array([171, 114])
# c = np.array([391, 328])  # -45 +-
# a = np.array([150, 313])
# b = np.array([356, 314])
# c = np.array([254, 62])  # -90 +-
a = np.array([2, 1])
b = np.array([4, 1])
c = np.array([3, 5])  # 90
angle = get_orientation(a, b, c)
print(angle)
