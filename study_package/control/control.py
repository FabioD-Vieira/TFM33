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
dist = np.sqrt(np.sum((circle_points - vessel_point)**2, axis=1))

sorted_indexes = np.argsort(dist)

plt.show()

