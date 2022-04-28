import math
from time import perf_counter_ns

import numpy as np
from matplotlib import pyplot as plt
from numpy import sin, cos, pi, linspace


radius = np.round(3 * 640 / 25).astype(int)


# draw arc
plt.xlim(0, 640)
plt.ylim(0, 480)
# plt.gca().set_aspect('equal')

Cx = 320
Cy = 240

arc_angles = linspace(0 * pi, 2 * pi, 20)
arc_xs = (radius * cos(arc_angles)) + Cx
arc_ys = (radius * sin(arc_angles)) + Cy

plt.plot(arc_xs, arc_ys, color='red', lw=3)

vessel_x = 250
vessel_y = 100

plt.plot([vessel_x], [vessel_y], marker="o", markersize=2, markeredgecolor="blue")

circle_points = np.stack((arc_xs, arc_ys), axis=1)

vessel_point = (vessel_x, vessel_y)
distances = np.sqrt(np.sum((circle_points - vessel_point)**2, axis=1))

min_error_index = np.argmin(distances)

points = np.roll(circle_points, -min_error_index, axis=0)
# errors = np.roll(distances, -min_error_index)

# for point in points:
#     plt.plot([Cx, point[0]], [Cy, point[1]], color='green', lw=3)
#
#     vector = point - [Cx, Cy]
#     theta = np.deg2rad(90)
#     # new_x = vector[0] * cos(theta) - vector[1] * sin(theta)
#     # new_y = vector[0] * sin(theta) + vector[1] * cos(theta)
#     rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
#     vector2 = np.dot(rot, vector)
#     new_x, new_y = vector2
#
#     plt.plot([point[0] - new_x, point[0] + new_x], [point[1] - new_y, point[1] + new_y], color='green', lw=3)
#
#     unit_vector_1 = vector / np.linalg.norm(vector)
#     unit_vector_2 = vector2 / np.linalg.norm(vector2)
#     dot_product = np.dot(unit_vector_1, unit_vector_2)
#     angle = np.arccos(dot_product)
#     print(np.rad2deg(angle))

point_index = 0
plt.plot([Cx, points[point_index][0]], [Cy, points[point_index][1]], color='green', lw=3)

vector = points[point_index] - [Cx, Cy]
theta = np.deg2rad(90)
# new_x = vector[0] * cos(theta) - vector[1] * sin(theta)
# new_y = vector[0] * sin(theta) + vector[1] * cos(theta)
rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
vector2 = np.dot(rot, vector)
new_x, new_y = vector2

plt.plot([points[point_index][0] - new_x, points[point_index][0] + new_x], [points[point_index][1] - new_y, points[point_index][1] + new_y], color='green', lw=3)

unit_vector_1 = vector / np.linalg.norm(vector)
unit_vector_2 = vector2 / np.linalg.norm(vector2)
dot_product = np.dot(unit_vector_1, unit_vector_2)
angle = np.arccos(dot_product)
print(np.rad2deg(angle))

threshold = 1

current_index = 0
previous_position_error = 0
previous_time = perf_counter_ns()

KP = 0.1
KD = 0.1

while True:
    # error = errors[current_index]
    target_position = points[current_index]

    position_error = target_position - vessel_point
    # print(target_vector)

    time_delta = perf_counter_ns() - previous_time

    derivative = (position_error - previous_position_error) / time_delta
    previous_position_error = position_error

    output = KP * position_error + KD * derivative
    # print("Output: " + str(output))

    # TODO code to update vessel position

    error_norm = math.sqrt(position_error[0]**2 + position_error[1]**2)
    # print("Error norm: " + str(error_norm))
    error_norm = 0  # to debug
    if error_norm < threshold:
        current_index += 1

    if current_index == len(points):
        # current_index = 0  # to reset behaviour
        break

    distances = np.sqrt(np.sum((circle_points - vessel_point) ** 2, axis=1))
    points = np.roll(circle_points, -min_error_index, axis=0)


plt.show()
