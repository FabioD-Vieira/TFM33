import math

import numpy as np
import pygame
from sklearn.neighbors import NearestNeighbors

from src.vessel_control.control.p_control import PControl


class Simulation:

    def __init__(self):
        self.__control = PControl(None, position_threshold=1, orientation_threshold=1)

        self.__background_colour = (255, 255, 255)
        (self.__width, self.__height) = (640, 480)

        self.__screen = pygame.display.set_mode((self.__width, self.__height))
        self.__screen.fill(self.__background_colour)

        self.__FPS = 30

        speed = 1  # m/s
        meter_per_pixel_x = 25 / 640
        meter_per_pixel_y = 10 / 480

        rotation = 10  # degree/s
        self.__rotation_per_frame = rotation / self.__FPS

        self.__point = (2 * (640/25), 9 * (480/10))
        self.__angle = 0
        self.__len = 10

        number_of_checkpoints = 1000

        init_line = (5, 5)
        line_angle = 0

        end_x = init_line[0] + math.cos(math.radians(-line_angle)) * self.__len
        end_y = init_line[1] + math.sin(math.radians(-line_angle)) * self.__len

        line_y = np.linspace(init_line[1], end_y, number_of_checkpoints)
        line_x = np.linspace(init_line[0], end_x, number_of_checkpoints)

        self.__line_points = np.stack((line_x * (640/25), line_y * (480/10)), axis=1)

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(self.__line_points)

        min_error_index = knn.kneighbors([self.__point], return_distance=False)[0][0]
        self.__target_y = self.__line_points[min_error_index][1]

        self.__pixels_per_frame_x = speed / meter_per_pixel_x / self.__FPS
        self.__pixels_per_frame_y = speed / meter_per_pixel_y / self.__FPS

        center_x, center_y = (100, 100)

        arc_angles = np.linspace(0.5 * np.pi, 0 * np.pi, number_of_checkpoints)

        # Add correct positions to each point
        radius = 50
        arc_xs = (radius * np.cos(arc_angles)) + center_x
        arc_ys = (radius * np.sin(arc_angles)) + center_y

        # Order counter clock wise
        # arc_ys = arc_ys[::-1]
        # arc_xs = arc_xs[::-1]

        self.circle = np.stack((arc_xs, arc_ys), axis=1)

    def __draw_vessel(self, x, y):
        # end_x = x + math.cos(math.radians(-self.__angle)) * self.__len
        # end_y = y + math.sin(math.radians(-self.__angle)) * self.__len

        pygame.draw.circle(self.__screen, (0, 0, 255), (self.__point[0], self.__point[1]), radius=10)
        # pygame.draw.line(self.__screen, (255, 255, 255), (self.__point[0], self.__point[1]), (end_x, end_y))

    def __draw_line(self):

        for p in self.circle:
            pygame.draw.circle(self.__screen, (0, 255, 0), p, radius=3)

        for point in self.__line_points:
            pygame.draw.circle(self.__screen, (0, 255, 0), point, radius=3)

    def __update_position(self, output_y):
        angle_limit = 90
        # self.__rotation_per_frame += self.__line_angle
        if output_y < 0:
            self.__angle += self.__rotation_per_frame
        elif output_y > 0:
            self.__angle -= self.__rotation_per_frame

        if self.__angle > angle_limit:
            self.__angle = angle_limit
        elif self.__angle < -angle_limit:
            self.__angle = -angle_limit

        new_x = self.__point[0] + math.cos(math.radians(-self.__angle)) * self.__pixels_per_frame_x
        new_y = self.__point[1] + math.sin(math.radians(-self.__angle)) * self.__pixels_per_frame_y + output_y

        self.__point = (new_x, new_y)

    def start(self):

        clock = pygame.time.Clock()
        pygame.display.flip()

        running = True
        while running:
            self.__screen.fill(self.__background_colour)

            self.__draw_vessel(self.__point[0], self.__point[1])
            self.__draw_line()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            y = self.__point[1]

            output_y = self.__control.get_y_output(y, self.__target_y)
            self.__update_position(output_y)

            pygame.display.update()
            clock.tick(self.__FPS)


if '__main__' == __name__:

    simulation = Simulation()
    simulation.start()
