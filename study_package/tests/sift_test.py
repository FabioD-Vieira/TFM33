import glob

import cv2
import numpy as np

from src.camera import Camera
from src.system import System


def corner_1(corner_img_1, corner_img_2):
    cv2.imshow("f", corner_img_1)
    cv2.imshow("f2", corner_img_2)

    kp1, des1 = sift.detectAndCompute(corner_img_1[:, :, 2], None)
    kp2, des2 = sift.detectAndCompute(corner_img_2[:, :, 2], None)

    matches = bf.knnMatch(des1, des2, k=2)

    ratio_dist = 0.5
    vector = np.array([0, 0], dtype='float')
    num = 0

    print(len(matches))
    for m, n in matches:
        # Apply ratio test
        if m.distance < ratio_dist * n.distance:
            (x1, y1) = kp1[m.queryIdx].pt
            (x2, y2) = kp2[m.trainIdx].pt

            vector += np.array([x2 - x1, y2 - y1])
            num += 1

    good = [[m] for m, n in matches if m.distance < ratio_dist * n.distance]
    img3 = cv2.drawMatchesKnn(corner_img_1, kp1, corner_img_2, kp2, good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("match", img3)

    vector = vector / num
    vector_x, vector_y = np.round(vector).astype(int)

    return vector_y, vector_x


image = cv2.imread("../../images/corners/POS4_5_4000_01.jpg")
image2 = cv2.imread("../../images/corners/POS4_5_4000_02.jpg")


camera_resolution = (640, 480)
pool_dim = (25, 10)

camera = Camera(camera_resolution, balance=0.9)

system = System(camera, pool_dim)

images = glob.glob('../../images/calibration/*.jpg')
system.calibrate_camera([cv2.imread(image_name) for image_name in images])

based_rotated = cv2.rotate(camera.un_distort(image), cv2.ROTATE_180)
img_rotated = cv2.rotate(camera.un_distort(image2), cv2.ROTATE_180)

sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=10)
bf = cv2.BFMatcher()

# i, j = (199, 192)
# based_rotated[i][j] = (255, 255, 255)
# vector_y, vector_x = corner_1(based_rotated[150:250, 150:250], img_rotated[150:250, 150:250])
# new_i, new_j = i + vector_y, j + vector_x
# print(new_i, new_j)
# img_rotated[new_i][new_j] = (255, 255, 255)

# i, j = (195, 454)
# based_rotated[i][j] = (255, 255, 255)
# vector_y, vector_x = corner_1(based_rotated[150:250, 400:500], img_rotated[150:250, 400:500])
# new_i, new_j = i + vector_y, j + vector_x
# print(new_i, new_j)
# img_rotated[new_i][new_j] = (255, 255, 255)

# i, j = (324, 37)
# based_rotated[i][j] = (255, 255, 255)
# vector_y, vector_x = corner_1(based_rotated[275:375, 0:100], img_rotated[275:375, 0:100])
# new_i, new_j = i + vector_y, j + vector_x
# print(new_i, new_j)
# img_rotated[new_i][new_j] = (255, 255, 255)

# i, j = (320, 638)
# based_rotated[i][j] = (255, 255, 255)
# vector_y, vector_x = corner_1(based_rotated[270:370, 540:640], img_rotated[270:370, 540:640])
# new_i, new_j = i + vector_y, j + vector_x
# print(new_i, new_j)
# img_rotated[new_i][new_j] = (255, 255, 255)

cv2.imshow("img", based_rotated)
cv2.imshow("img2", img_rotated)

def print_coordinates(event, x_coord, y_coord, _, _1):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(y_coord, x_coord)


cv2.setMouseCallback("img", print_coordinates)


# # matches = sorted(matches, key=lambda x: x.distance)
#

# img = cv2.drawKeypoints(corner1[:, :, 2], kp, corner1)
# cv2.imshow('sift_keypoints.jpg', corner1)

cv2.waitKey(0)
cv2.destroyAllWindows()
