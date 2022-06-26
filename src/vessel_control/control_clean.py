import math

import numpy as np
# from matplotlib import pyplot as plt

from sklearn.neighbors import NearestNeighbors


# Variables
pool_dim = (25, 10)
AV_power = 20
AC_max_power = 20


# To plot line/circle/vessel
# plt.xlim(0, pool_dim[0])
# plt.ylim(pool_dim[1], 0)
# plt.gca().set_aspect('equal')


# Variables to create line/circle
number_of_points = 1000


# Create line
target_angle = 0
line_size = 10

line_start_x, line_start_y = (0, 5)

line_end_x = line_start_x + math.cos(math.radians(target_angle)) * 25
line_end_y = line_start_y + math.sin(math.radians(target_angle)) * 10

line_x = np.linspace(line_start_x, line_end_x, number_of_points)
line_y = np.linspace(line_start_y, line_end_y, number_of_points)

# plt.plot(line_x, line_y, color='red', lw=3)

line = np.stack((line_x, line_y), axis=1)


# Read data
x, y = 3, 5
angle = 0

# plt.plot([x], [y], marker="o", markersize=2, markeredgecolor="blue")


# Find the closest point
knn = NearestNeighbors(n_neighbors=1)
knn.fit(line)
# knn.fit(concave_arc)
# knn.fit(convex_arc)

min_error_index = knn.kneighbors([(x, y)], return_distance=False)[0][0]
current_index = min_error_index

# Gains
K_angle = 0.1
KP_position = 0.1

while True:

    # Read data
    # x, y = (10, 5.5)
    # angle = -5

    min_error_index = knn.kneighbors([(x, y)], return_distance=False)[0][0]
    target_point = line[min_error_index]

    if x < 2 or x > 10 or y < 2 or y > 8:
        AV = 0
        AC = 0

        # send info to vessel
        print("Stopping...")
        continue

    target = (round(target_point[0], 2), round(target_point[1], 2))

    error_vector = (target[0] - x, target[1] - y)
    d = math.sqrt((error_vector[0] ** 2) + (error_vector[1] ** 2))

    AV = AV_power

    orientation_diff = abs(target_angle - angle)

    if orientation_diff > 40:
        continue

    D = d + K_angle * orientation_diff
    AC = KP_position * D

    if error_vector[1] > 0:
        AC = -AC

    if AC > AC_max_power:
        AC = AC_max_power

    elif AC < -AC_max_power:
        AC = -AC_max_power

    left_engine = AV - AC
    right_engine = AV + AC

    print(left_engine, right_engine)

    break

# plt.show()
