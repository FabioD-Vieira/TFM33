import math

import numpy as np
import pygame
from sklearn.neighbors import NearestNeighbors

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


# Create line
number_of_checkpoints = 1000

line_init = (0, 5)
line_angle = 0

line_end_x = line_init[0] + math.cos(math.radians(line_angle)) * pool_dim[0]
line_end_y = line_init[1] + math.sin(math.radians(line_angle)) * pool_dim[1]

line_x = np.linspace(line_init[0], line_end_x, number_of_checkpoints)
line_y = np.linspace(line_init[1], line_end_y, number_of_checkpoints)

line = np.stack((line_x, line_y), axis=1)


def draw_line():
    for point in line:
        pygame.draw.circle(screen, (0, 255, 0), convert_point(point), radius=1)


vessel_pos = (2, 5)
vessel_orientation = 0


knn = NearestNeighbors(n_neighbors=1)
knn.fit(line)

min_error_index = knn.kneighbors([vessel_pos], return_distance=False)[0][0]
target_point = line[min_error_index]

target = (round(target_point[0], 2), round(target_point[1], 2))


def update_vessel(pos, orientation, left_eng, right_eng):
    diff = left_eng - right_eng

    orientation += diff

    speed_per_frame = speed / FPS

    new_x = pos[0] + math.cos(math.radians(orientation)) * speed_per_frame
    new_y = pos[1] + math.sin(math.radians(orientation)) * speed_per_frame

    return (new_x, new_y), orientation


K_orientation = 0.1
KP_position = 0.1

AV_power = 20
AC_max_power = 20

clock = pygame.time.Clock()
running = True
paused = False
while running:
    clock.tick(FPS)
    screen.fill((0, 0, 255))

    draw_vessel(vessel_pos, vessel_orientation)
    draw_line()

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                paused = not paused

    if not paused:

        min_error_index = knn.kneighbors([vessel_pos], return_distance=False)[0][0]
        target_point = line[min_error_index]
        target_angle = line_angle

        target = (round(target_point[0], 2), round(target_point[1], 2))

        error_vector = (target[0] - vessel_pos[0], target[1] - vessel_pos[1])
        d = math.sqrt((error_vector[0] ** 2) + (error_vector[1] ** 2))

        orientation_diff = abs(target_angle - vessel_orientation)

        D = d + K_orientation * orientation_diff
        AC = KP_position * D

        AV = AV_power

        if error_vector[1] > 0:
            AC = -AC

        left_engine = AV - AC
        right_engine = AV + AC

        vessel_pos, vessel_orientation = update_vessel(vessel_pos, vessel_orientation, left_engine, right_engine)
