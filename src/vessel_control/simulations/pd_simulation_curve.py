import math

import numpy as np
import pygame
from sklearn.neighbors import NearestNeighbors

pygame.font.init()
font = pygame.font.Font('freesansbold.ttf', 28)
text = font.render('PD Control', True, (255, 255, 255))

(width, height) = (640, 480)
screen = pygame.display.set_mode((width, height))
FPS = 30

pool_dim = (25, 10)

speed = 1  # m/s


def convert_point(point):
    return point[0] * (width / pool_dim[0]), point[1] * (height / pool_dim[1])


def draw_vessel(pos, orientation):
    radius = 10

    x, y = convert_point(pos)

    end_x = x + math.cos(math.radians(orientation)) * radius
    end_y = y + math.sin(math.radians(orientation)) * radius

    pygame.draw.circle(screen, (255, 255, 255), (x, y), radius=radius)
    pygame.draw.line(screen, (0, 0, 0), (x, y), (end_x, end_y))


# Create curve
number_of_checkpoints = 1000

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


def draw_curve():
    for point in curve:
        pygame.draw.circle(screen, (0, 255, 0), convert_point(point), radius=1)


vessel_pos = (1, 4)
vessel_orientation = 0


knn = NearestNeighbors(n_neighbors=1)
knn.fit(curve)

min_error_index = knn.kneighbors([vessel_pos], return_distance=False)[0][0]
target_point = curve[min_error_index]

target = (round(target_point[0], 2), round(target_point[1], 2))


def update_vessel(pos, orientation, left_eng, right_eng):
    diff = left_eng - right_eng

    orientation += diff

    speed_per_frame = speed / FPS

    new_x = pos[0] + math.cos(math.radians(orientation)) * speed_per_frame
    new_y = pos[1] + math.sin(math.radians(orientation)) * speed_per_frame

    return (new_x, new_y), orientation


K_orientation = 0.1
KP_position = 0.08
KD_position = 0.08

AV_power = 20
AC_max_power = 20

time_delta = 0.01
previous_d = 0

clock = pygame.time.Clock()
running = True
paused = False
while running:
    clock.tick(FPS)
    screen.fill((0, 0, 255))
    screen.blit(text, (0, 0))

    draw_vessel(vessel_pos, vessel_orientation)
    draw_curve()

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                paused = not paused

    if not paused:

        min_error_index = knn.kneighbors([vessel_pos], return_distance=False)[0][0]
        target_point = curve[min_error_index]
        target_angle = curve_angles[min_error_index]

        target = (round(target_point[0], 2), round(target_point[1], 2))

        error_vector = (target[0] - vessel_pos[0], target[1] - vessel_pos[1])
        d = math.sqrt((error_vector[0] ** 2) + (error_vector[1] ** 2))

        orientation_diff = abs(target_angle - vessel_orientation)
        # orientation_diff = 0

        D = d + K_orientation * orientation_diff

        position_derivative = (d - previous_d) / time_delta
        previous_d = d

        AC = KP_position * D + KD_position * position_derivative

        AV = AV_power

        if error_vector[1] > 0:
            AC = -AC

        left_engine = AV - AC
        right_engine = AV + AC

        vessel_pos, vessel_orientation = update_vessel(vessel_pos, vessel_orientation, left_engine, right_engine)
