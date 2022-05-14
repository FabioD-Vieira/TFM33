import math

import numpy as np
import pygame

from src.vessel_control.control.p_control import PControl
from src.vessel_control.shape.circle import Circle


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

        self.__line_y = 5 * (480/10)
        self.__line_x = np.array([5, 8, 11, 14, 17, 20]) * (640/25)

        self.__pixels_per_frame_x = speed / meter_per_pixel_x / self.__FPS
        self.__pixels_per_frame_y = speed / meter_per_pixel_y / self.__FPS

    def __draw_vessel(self, x, y):
        end_x = x + math.cos(math.radians(-self.__angle)) * self.__len
        end_y = y + math.sin(math.radians(-self.__angle)) * self.__len

        pygame.draw.circle(self.__screen, (0, 0, 255), (self.__point[0], self.__point[1]), radius=10)
        # pygame.draw.line(self.__screen, (255, 255, 255), (self.__point[0], self.__point[1]), (end_x, end_y))

    def __draw_line(self):

        for x in self.__line_x:
            pygame.draw.circle(self.__screen, (0, 255, 0), (x, self.__line_y), radius=3)

    def __update_position(self, output_y):
        if output_y < 0:
            self.__angle += self.__rotation_per_frame
        elif output_y > 0:
            self.__angle -= self.__rotation_per_frame

        if self.__angle > 90:
            self.__angle = 90
        elif self.__angle < -90:
            self.__angle = -90

        new_x = self.__point[0] + math.cos(math.radians(-self.__angle)) * self.__pixels_per_frame_x
        new_y = self.__point[1] + math.sin(math.radians(-self.__angle)) * self.__pixels_per_frame_y + output_y

        self.__point = (new_x, new_y)

    def start(self):

        # current_index = 0

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

            target_y = self.__line_y
            y = self.__point[1]

            output_y = self.__control.get_y_output(y, target_y)
            self.__update_position(output_y)

            pygame.display.update()
            clock.tick(self.__FPS)


if '__main__' == __name__:

    simulation = Simulation()
    simulation.start()
