import math

import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import NearestNeighbors


# Variables
pool_dim = (25, 10)
AV_power = 20
AC_max_power = 20


# To plot line/circle/vessel
plt.xlim(0, pool_dim[0])
plt.ylim(pool_dim[1], 0)
plt.gca().set_aspect('equal')


# Variables to create line/circle
number_of_points = 1000

radius = 3

# Matrix to rotate 90 degrees
theta = np.deg2rad(90)
rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


# Create concave curve
arc_angles = np.linspace(np.pi, 1.5 * np.pi, number_of_points)

arc_center = (11, 6)
arc_xs = (radius * np.cos(arc_angles)) + arc_center[0]
arc_ys = (radius * np.sin(arc_angles)) + arc_center[1]

# plt.plot(arc_xs, arc_ys, color='red', lw=3)

concave_arc = np.stack((arc_xs, arc_ys), axis=1)

# Calculate vectors to the center of the circle
vectors_to_center = concave_arc - [arc_center[0], arc_center[1]]

# Calculate tangent vector of each circle point by rotating 90º each vector to the center
concave_points_orientations = np.zeros(len(concave_arc))
for i in range(len(vectors_to_center)):
    vector = np.dot(rot, vectors_to_center[i])
    concave_points_orientations[i] = math.degrees(math.atan2(vector[1], vector[0]))

# Create convex curve
arc_angles = np.linspace(0.5 * np.pi, 0 * np.pi, number_of_points)

arc_center = (7, 2)
arc_xs = (radius * np.cos(arc_angles)) + arc_center[0]
arc_ys = (radius * np.sin(arc_angles)) + arc_center[1]

plt.plot(arc_xs, arc_ys, color='red', lw=3)

convex_arc = np.stack((arc_xs, arc_ys), axis=1)

# Calculate vectors to the center of the circle
vectors_to_center = convex_arc - [arc_center[0], arc_center[1]]

# Calculate tangent vector of each circle point by rotating 90º each vector to the center
convex_points_orientations = np.zeros(len(convex_arc))
for i in range(len(vectors_to_center)):
    vector = np.dot(rot, vectors_to_center[i])
    convex_points_orientations[i] = math.degrees(math.atan2(vector[1], vector[0])) - 180

# Create line
target_angle = 5
line_size = 10

line_start_x, line_start_y = (5, 5)

line_end_x = line_start_x + math.cos(math.radians(target_angle)) * line_size
line_end_y = line_start_y + math.sin(math.radians(target_angle)) * line_size

line_x = np.linspace(line_start_x, line_end_x, number_of_points)
line_y = np.linspace(line_start_y, line_end_y, number_of_points)

line = np.stack((line_x, line_y), axis=1)


# Read data
x, y = (10, 5.5)
# angle = -5
plt.plot([x], [y], marker="o", markersize=2, markeredgecolor="blue")

# Find the closest point
knn = NearestNeighbors(n_neighbors=1)
knn.fit(line)
# knn.fit(concave_arc)
# knn.fit(convex_arc)

min_error_index = knn.kneighbors([(x, y)], return_distance=False)[0][0]
target_point = line[min_error_index]

# target_point = concave_arc[min_error_index]
# target_angle = concave_points_orientations[min_error_index]

# target_point = convex_arc[min_error_index]
# target_angle = convex_points_orientations[min_error_index]

# print(min_error_index)
# print(target_point, target_angle)

# Gains
K_angle = 0.1
KP_position = 0.1

while True:

    # Read data
    x, y = (2, 7)
    angle = -5

    if x < 2 or x > 10 or y < 2 or y > 8:
        AV = 0
        AC = 0

        # send info to vessel
        print("Stopping...")
        break

    AV = AV_power
    angle_diff = target_angle - angle

    d = math.sqrt((target_point[0] - x)**2 + (target_point[1] - y)**2)
    D = d + K_angle * angle_diff

    AC = KP_position * D

    if AC > AC_max_power:
        AC = AC_max_power

    elif AC < -AC_max_power:
        AC = -AC_max_power

    left_engine = AV + AC
    right_engine = AV - AC

    print(left_engine, right_engine)

    break

plt.show()
