import math
from time import perf_counter_ns

import numpy as np
from matplotlib import pyplot as plt
from numpy import sin, cos, pi, linspace

ratio = 10 / 25

width = 640
height = 480
height = np.round(width * ratio).astype(int)

vessel_x = 250
vessel_y = 100
vessel_orientation = 45

radius = np.round(3 * width / 25).astype(int)

# draw arc
plt.xlim(0, 640)
plt.ylim(0, 480)
# plt.gca().set_aspect('equal')

Cx = width / 2
Cy = height / 2

arc_angles = linspace(0 * pi, 2 * pi, 20)
arc_xs = (radius * cos(arc_angles)) + Cx
arc_ys = (radius * sin(arc_angles)) + Cy

plt.plot(arc_xs, arc_ys, color='red', lw=3)
plt.plot([vessel_x], [vessel_y], marker="o", markersize=2, markeredgecolor="blue")

circle_points = np.stack((arc_xs, arc_ys), axis=1)

vessel_point = (vessel_x, vessel_y)
distances = np.sqrt(np.sum((circle_points - vessel_point)**2, axis=1))

min_error_index = np.argmin(distances)

points = np.roll(circle_points, -min_error_index, axis=0)
# errors = np.roll(distances, -min_error_index)


theta = np.deg2rad(90)
rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
perpendicular_vectors = points - [Cx, Cy]

target_orientations = np.zeros(len(points))
for i in range(len(perpendicular_vectors)):
    vector = np.dot(rot, perpendicular_vectors[i])
    target_orientations[i] = -math.degrees(math.atan2(vector[1], vector[0]))

debug = 0
plt.plot([Cx, points[debug][0]], [Cy, points[debug][1]], color='green', lw=3)

vector2 = np.dot(rot, perpendicular_vectors[debug])

angle = -math.degrees(math.atan2(vector2[1], vector2[0]))
# print(angle)

plt.plot([points[debug][0] - vector2[0], points[debug][0] + vector2[0]],
         [points[debug][1] - vector2[1], points[debug][1] + vector2[1]], color='green', lw=3)


position_threshold = 1
orientation_threshold = 1

previous_position_error = 0
previous_orientation_error = 0

previous_time = perf_counter_ns()

position_KP = 0.1
position_KD = 0.1

orientation_KP = 0.1
orientation_KD = 0.1

current_index = 0
while True:
    time_delta = perf_counter_ns() - previous_time
    previous_time = time_delta

    # error = errors[current_index]
    target_position = points[current_index]
    position_error = target_position - vessel_point

    position_derivative = (position_error - previous_position_error) / time_delta
    previous_position_error = position_error

    position_output = position_KP * position_error + position_KD * position_derivative

    # TODO code to update vessel position

    target_orientation = target_orientations[current_index]
    orientation_error = target_orientation - vessel_orientation

    orientation_derivative = (orientation_error - previous_orientation_error) / time_delta
    previous_orientation_error = orientation_error

    angle_output = orientation_KP * orientation_error + orientation_KD * orientation_derivative

    # TODO code to update vessel orientation

    position_error_norm = math.sqrt(position_error[0]**2 + position_error[1]**2)

    position_error_norm = 0  # to debug
    orientation_error = 0  # to debug

    if position_error_norm < position_threshold and orientation_error < orientation_threshold:
        current_index += 1

    if current_index == len(points):
        # current_index = 0  # to reset behaviour
        break

plt.show()
