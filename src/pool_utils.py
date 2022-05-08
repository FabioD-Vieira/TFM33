import math

import cv2

RED = 2
GREEN = 1
BLUE = 0


class PoolUtils:

    @staticmethod
    def __get_points(image):

        # Find RED color in image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_image, (0, 80, 20), (20, 255, 255))
        # red_mask = cv2.inRange(hsv_image, (170, 50, 20), (190, 255, 255))
        # red_mask = cv2.bitwise_or(mask_1, mask_2)

        # Find LIGHT in image
        image_channel = image[:, :, RED]
        _, light_mask = cv2.threshold(image_channel, 100, 255, cv2.THRESH_BINARY)

        # Join both masks and find LED contours
        mask = cv2.bitwise_and(red_mask, light_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detecting a single point because test images only have a single LED rectangle
        point = [sum(contour) / len(contour) for contour in contours]
        return point[0][0]

        # assert len(contours) == 3, "No vessel detected"
        #
        # point_a, point_b, point_c = [sum(contour) / len(contour) for contour in contours]

        # return point_a[0], point_b[0], point_c[0]

    @staticmethod
    def __find_back_and_front(point_a, point_b, point_c):

        # Distance between each point
        distance_a_b = math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
        distance_a_c = math.sqrt((point_a[0] - point_c[0]) ** 2 + (point_a[1] - point_c[1]) ** 2)
        distance_c_b = math.sqrt((point_c[0] - point_b[0]) ** 2 + (point_c[1] - point_b[1]) ** 2)

        # Two closest points belong to the front
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

    @staticmethod
    def get_vessel_info(image):

        # Uncomment for 3 LED rectangle images
        """point_a, point_b, point_c = __get_points(image)

        # -90º
        # point_a = np.array([10, 10])
        # point_b = np.array([30, 10])
        # point_c = np.array([20, 50])

        back, front = __find_back_and_front(point_a, point_b, point_c)
        x, y = (front + back) / 2

        return x, y, math.degrees(math.atan2(front[1] - back[1], front[0] - back[0]))"""

        # Test with only one LED rectangle
        point = PoolUtils.__get_points(image)
        return point[0], point[1], 0
