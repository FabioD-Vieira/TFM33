import math

import pygame


class Simulation:
    def __init__(self, text, trajectory, width=640, height=256, pool_dim=(25, 10)):
        pygame.font.init()
        font = pygame.font.Font('freesansbold.ttf', 28)
        self.__text = font.render(text, True, (255, 255, 255))

        self.__trajectory = trajectory

        (self.__width, self.__height) = (width, height)
        self.__pool_dim = pool_dim

        self.__screen = pygame.display.set_mode((width, height))

        self.__FPS = 30
        self.__clock = pygame.time.Clock()

        self.__speed = 1  # m/s
        self.__rotation = 30  # degree/s

        self.__running = True

        self.__old_positions = []

    def draw(self, position, orientation, AV, AC):
        self.__clock.tick(self.__FPS)

        self.__screen.fill((0, 0, 255))
        self.__screen.blit(self.__text, (0, 0))

        self.__draw_vessel(position, orientation)
        self.__draw_trajectory()

        self.__draw_old_positions()

        pygame.display.update()

        left_engine = AV + AC
        right_engine = AV - AC

        position, orientation = self.__update_vessel(position, orientation, left_engine, right_engine)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__running = False

        return self.__running, position, orientation

    def __draw_vessel(self, pos, orientation):
        radius = 10

        x, y = self.__convert_point(pos)

        end_x = x + math.cos(math.radians(orientation)) * radius
        end_y = y + math.sin(math.radians(orientation)) * radius

        pygame.draw.circle(self.__screen, (255, 255, 255), (x, y), radius=radius)
        pygame.draw.line(self.__screen, (0, 0, 0), (x, y), (end_x, end_y))

    def __draw_old_positions(self):
        for point in self.__old_positions:
            pygame.draw.circle(self.__screen, (255, 140, 0), self.__convert_point(point), radius=1)

    def __draw_trajectory(self):
        for point in self.__trajectory:
            pygame.draw.circle(self.__screen, (0, 255, 0), self.__convert_point(point), radius=1)

    def __convert_point(self, point):
        return point[0] * (self.__width / self.__pool_dim[0]), point[1] * (self.__height / self.__pool_dim[1])

    def __update_vessel(self, pos, orientation, left_eng, right_eng):

        if left_eng > right_eng:
            s = 1
        elif right_eng > left_eng:
            s = -1
        else:
            s = 0

        rotation_per_frame = self.__rotation / self.__FPS
        orientation += s * rotation_per_frame

        speed_per_frame = self.__speed / self.__FPS

        new_x = pos[0] + math.cos(math.radians(orientation)) * speed_per_frame
        new_y = pos[1] + math.sin(math.radians(orientation)) * speed_per_frame

        self.__old_positions.append((new_x, new_y))

        return (new_x, new_y), orientation
