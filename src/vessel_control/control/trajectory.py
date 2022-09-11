import math

import numpy as np

number_of_checkpoints = 1000


def create_curve():

    radius_x = 15
    radius_y = 7

    arc_center = (12.5, -1)

    arc_angles = np.linspace(1 * np.pi, 0 * np.pi, number_of_checkpoints)

    arc_xs = (radius_x * np.cos(arc_angles)) + arc_center[0]
    arc_ys = (radius_y * np.sin(arc_angles)) + arc_center[1]

    curve = np.stack((arc_xs, arc_ys), axis=1)

    # Calculate vectors to the center of the circle
    vectors_to_center = curve - [arc_center[0], arc_center[1]]

    # Matrix to rotate 90 degrees
    theta = np.deg2rad(90)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Calculate tangent vector of each circle point by rotating 90º each vector to the center
    curve_angles = np.zeros(len(curve))
    for i in range(len(vectors_to_center)):
        vector = np.dot(rot, vectors_to_center[i])
        curve_angles[i] = math.degrees(math.atan2(vector[0], vector[1])) + 90

    return curve, curve_angles


def create_line(pool_dim):

    line_init = (0, 5)
    line_angle = 10

    line_end_x = line_init[0] + math.cos(math.radians(line_angle)) * pool_dim[0]
    line_end_y = line_init[1] + math.sin(math.radians(line_angle)) * pool_dim[1]

    line_x = np.linspace(line_init[0], line_end_x, number_of_checkpoints)
    line_y = np.linspace(line_init[1], line_end_y, number_of_checkpoints)

    line = np.stack((line_x, line_y), axis=1)
    return line, np.full(len(line), line_angle)
