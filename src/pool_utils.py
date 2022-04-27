import math

import cv2

RED = 2
GREEN = 1
BLUE = 0


def get_points(image):

    # RED
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(hsv_image, (0, 50, 20), (5, 255, 255))
    mask_2 = cv2.inRange(hsv_image, (170, 50, 20), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask_1, mask_2)

    # Light
    image_channel = image[:, :, RED]
    _, light_mask = cv2.threshold(image_channel, 80, 255, cv2.THRESH_BINARY)

    final_red_mask = cv2.bitwise_and(red_mask, light_mask)

    contours, hierarchy = cv2.findContours(final_red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    point_a, point_b, point_c = [sum(contour) / len(contour) for contour in contours]

    return point_a[0], point_b[0], point_c[0]


def get_coordinates(point_a, point_b, point_c):
    return sum([point_a, point_b, point_c]) / 3


def get_orientation(point_a, point_b, point_c, x, y):

    distance_a_b = math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
    distance_a_c = math.sqrt((point_a[0] - point_c[0]) ** 2 + (point_a[1] - point_c[1]) ** 2)
    distance_c_b = math.sqrt((point_c[0] - point_b[0]) ** 2 + (point_c[1] - point_b[1]) ** 2)

    if distance_a_b < distance_a_c and distance_a_b < distance_c_b:
        back = point_c
    elif distance_a_c < distance_a_b and distance_a_c < distance_c_b:
        back = point_b
    else:
        back = point_a

    return math.degrees(math.atan2(y - back[1], x - back[0])) + 90


def get_vessel_info(image):

    point_a, point_b, point_c = get_points(image)

    # point_a = np.array([200, 10])
    # point_b = np.array([200, 50])
    # point_c = np.array([50, 30])
    #
    # image[point_a[0]][point_a[1]] = (255, 255, 255)
    # image[point_b[0]][point_b[1]] = (255, 255, 255)
    # image[point_c[0]][point_c[1]] = (255, 255, 255)

    x, y = get_coordinates(point_a, point_b, point_c)
    angle = get_orientation(point_a, point_b, point_c, x, y)

    return x, y, angle
