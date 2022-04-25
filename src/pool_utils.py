import math

import cv2
import numpy as np

RED = 2
GREEN = 1
BLUE = 0


def centroid(hsv_image, image, channel, limits):
    image_channel = image[:, :, channel]

    _, light_mask = cv2.threshold(image_channel, 150, 255, cv2.THRESH_BINARY)

    previous_mask = None
    mask = None
    for limit in limits:
        lower_limit, upper_limit = limit

        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
        previous_mask = cv2.bitwise_or(previous_mask, mask) if previous_mask is not None else mask

    mask = previous_mask if previous_mask is not None else mask
    final_mask = cv2.bitwise_and(mask, light_mask)

    pixels = np.where(final_mask == 255)

    if len(pixels[0]) == 0:
        return

    return sum(np.stack(pixels, axis=1)) / len(pixels[0])


def get_location_in_image(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_limits = [[(0, 50, 20), (5, 255, 255)], [(175, 50, 20), (180, 255, 255)]]
    back_point = centroid(hsv_image, image, RED, red_limits)

    green_limits = [[(40, 40, 40), (70, 255, 255)]]
    front_point = centroid(hsv_image, image, GREEN, green_limits)

    return back_point, front_point


def get_orientation(back_point, front_point):

    angle_radians = math.atan2(front_point[1] - back_point[1], front_point[0] - back_point[0])
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


# red_channel = image[:, :, 2]
# _, red_light_mask = cv2.threshold(red_channel, 150, 255, cv2.THRESH_BINARY)
#
# red_mask1 = cv2.inRange(hsv_image, (0, 50, 20), (5, 255, 255))
# red_mask2 = cv2.inRange(hsv_image, (175, 50, 20), (180, 255, 255))
#
# red_mask = cv2.bitwise_or(red_mask1, red_mask2)
# final_red_mask = cv2.bitwise_and(red_mask, red_light_mask)
#
# green_channel = image[:, :, 1]
# _, green_light_mask = cv2.threshold(green_channel, 150, 255, cv2.THRESH_BINARY)
#
# green_mask = cv2.inRange(hsv_image, (40, 40, 40), (70, 255, 255))
# final_green_mask = cv2.bitwise_and(green_mask, green_light_mask)
