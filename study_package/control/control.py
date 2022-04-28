from time import perf_counter_ns

import numpy as np
from matplotlib import pyplot as plt
from numpy import sin, cos, pi, linspace


radius = np.round(3 * 640 / 25).astype(int)


# draw arc
plt.xlim(0, 640)
plt.ylim(0, 480)
plt.gca().set_aspect('equal')

arc_angles = linspace(0 * pi, 2 * pi, 20)
arc_xs = (radius * cos(arc_angles)) + 320
arc_ys = (radius * sin(arc_angles)) + 240

plt.plot(arc_xs, arc_ys, color='red', lw=3)

vessel_x = 250
vessel_y = 100

plt.plot([vessel_x], [vessel_y], marker="o", markersize=2, markeredgecolor="blue")

circle_points = np.stack((arc_xs, arc_ys), axis=1)

vessel_point = (vessel_x, vessel_y)
distances = np.sqrt(np.sum((circle_points - vessel_point)**2, axis=1))

min_error_index = np.argmin(distances)

points = np.roll(circle_points, -min_error_index, axis=0)
errors = np.roll(distances, -min_error_index)

threshold = 1

current_index = 0
previousError = 0
previousTime = perf_counter_ns()

KP = 0.1
KD = 0.1

while True:
    error = errors[current_index]
    target = points[current_index]

    target_vector = target - vessel_point
    print(target_vector)

    timeDelta = perf_counter_ns() - previousTime

    derivative = (error - previousError) / timeDelta
    previousError = error

    output = KP * error + KD * derivative
    print(output)

    # vessel updated in the meantime

    error = 0
    if error < threshold:
        current_index += 1

    if current_index == len(errors):
        break

    distances = np.sqrt(np.sum((circle_points - vessel_point) ** 2, axis=1))
    errors = np.roll(distances, -min_error_index)


plt.show()
