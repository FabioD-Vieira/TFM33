import math

import cv2

RED = 2
GREEN = 1
BLUE = 0


def __get_points(image):

    # RED
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mask_1 = cv2.inRange(hsv_image, (0, 50, 20), (5, 255, 255))
    red_mask = cv2.inRange(hsv_image, (120, 50, 20), (175, 255, 255))
    # red_mask = cv2.bitwise_or(mask_1, mask_2)
    # cv2.imshow("red", red_mask)

    # Light
    image_channel = image[:, :, RED]
    # cv2.imshow("image_channel", image_channel)
    _, light_mask = cv2.threshold(image_channel, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow("light_mask", light_mask)

    # return np.array([0, 0]), np.array([1, 1]), np.array([2, 2])

    final_red_mask = cv2.bitwise_and(red_mask, light_mask)

    contours, hierarchy = cv2.findContours(final_red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 3:
        raise Exception("No vessel detected")

    point_a, point_b, point_c = [sum(contour) / len(contour) for contour in contours]
    return point_a[0], point_b[0], point_c[0]


def __find_back_and_front(point_a, point_b, point_c):
    distance_a_b = math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
    distance_a_c = math.sqrt((point_a[0] - point_c[0]) ** 2 + (point_a[1] - point_c[1]) ** 2)
    distance_c_b = math.sqrt((point_c[0] - point_b[0]) ** 2 + (point_c[1] - point_b[1]) ** 2)

    if distance_a_b < distance_a_c and distance_a_b < distance_c_b:
        front = point_a, point_b
        back = point_c
    elif distance_a_c < distance_a_b and distance_a_c < distance_c_b:
        front = point_a, point_c
        back = point_b
    else:
        front = point_b, point_c
        back = point_a

    front = sum(front) / 2
    return back, front


def get_vessel_info(image):
    point_a, point_b, point_c = __get_points(image)

    # -90º
    # point_a = np.array([10, 10])
    # point_b = np.array([30, 10])
    # point_c = np.array([20, 50])

    back, front = __find_back_and_front(point_a, point_b, point_c)
    x, y = (front + back) / 2

    return x, y, math.degrees(math.atan2(front[1] - back[1], front[0] - back[0]))
