import glob
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.image_processing.process.process import Process
from src.image_processing.setup.camera import Camera
from src.image_processing.setup.setup import Setup


def get_coord(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv_image, (0, 80, 20), (20, 255, 255))
    mask2 = cv2.inRange(hsv_image, (175, 50, 20), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)

    image_channel = image[:, :, 2]
    _, light_mask = cv2.threshold(image_channel, 100, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_and(red_mask, light_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.float32))
    cv2.imshow("mask", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return [sum(contour) / len(contour) for contour in contours]


def find_back_and_front(point_a, point_b , point_c):

    # Distance between each point
    distance_a_b = math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
    distance_a_c = math.sqrt((point_a[0] - point_c[0]) ** 2 + (point_a[1] - point_c[1]) ** 2)
    distance_c_b = math.sqrt((point_c[0] - point_b[0]) ** 2 + (point_c[1] - point_b[1]) ** 2)

    # Two closest points belong to the front
    if distance_a_b < distance_a_c and distance_a_b < distance_c_b:
        front = np.array([point_a, point_b])
        back = point_c
    elif distance_a_c < distance_a_b and distance_a_c < distance_c_b:
        front = np.array([point_a, point_c])
        back = point_b
    else:
        front = np.array([point_b, point_c])
        back = point_a

    front = sum(front) / 2
    return back, front


def get_coord_angle(back_point, front_point):
    x, y = (front_point + back_point) / 2
    angle = math.degrees(math.atan2(front_point[1] - back_point[1], front_point[0] - back_point[0]))

    return x, y, angle


def process(image):

    points_coord = get_coord(image)

    if len(points_coord) != 3:
        return None, None

    a, b, c = points_coord
    a, b, c = a[0], b[0], c[0]
    back, front = find_back_and_front(a, b, c)
    x, y, angle = get_coord_angle(back, front)
    return (x, y), angle


cap = cv2.VideoCapture("../../../images/videos/vid01.h264")

camera_resolution = (640, 480)
pool_dim = (25, 10)

camera = Camera(camera_resolution, balance=0.9)
setup = Setup(camera, pool_dim)

images = glob.glob('../../../images/calibration/*.jpg')
setup.calibrate_camera([cv2.imread(image_name) for image_name in images])

base_image = cv2.imread("../../../images/corners/POS4_5_4000_01.jpg")
img = cv2.imread("../../../images/corners/img4LUT.jpg")  # This image is read from the camera

setup.calculate_homography_matrix(base_image, img)

process_class = Process(None, pool_dim)

points = []
angles = []

list_of_points = []
list_of_angles = []

i = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("frame", frame)

    if True:

        reprojected = setup.no_lut_process(frame)
        cv2.imshow("re", reprojected)

        point, angle = process(reprojected)

        if point is not None:
            points.append(point)
            angles.append(angle)
        else:
            list_of_points.append(points)
            list_of_angles.append(angles)
            points = []
            angles = []

    i+=1
    cv2.waitKey(1)
cv2.destroyAllWindows()

list_of_points.append(points)
list_of_angles.append(angles)

print("points=np.array(" + str(list_of_points) + ")")
print("angles=np.array(" + str(list_of_angles) + ")")
