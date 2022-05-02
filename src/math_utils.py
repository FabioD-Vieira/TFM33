import numpy as np


def calculate_intersection_point(lines):

    if len(lines) != 2:
        raise Exception("Pool corners not detected")

    ax, ay, bx, by = lines[0][0]
    m1 = (by - ay) / (bx - ax)
    b1 = ay - (m1 * ax)

    cx, cy, dx, dy = lines[1][0]
    m2 = (dy - cy) / (dx - cx)
    b2 = cy - (m2 * cx)

    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return np.round(x).astype(int), np.round(y).astype(int)
