import glob

import cv2
import numpy as np

from src.camera import Camera
from src.system import System

system = System(camera_resolution=(640, 480), camera_balance=0.9)
camera = Camera(res=(640, 480), balance=0.9)

images = glob.glob('../../images/calibration/*.jpg')
camera.calibrate(images)  # Called only once to calibrate

img = cv2.imread("../../images/pool/pool.jpeg")
# cv2.imshow("Original", img)

undistorted = camera.un_distort(img)
# cv2.imshow("Undistorted", undistorted)

# Rotation cant be done before calibration
undistorted = cv2.rotate(undistorted, cv2.ROTATE_180)
# cv2.imshow("Rotated", undistorted)

def check_intersection(a, b):
    return a[0] * b[1] - a[1] * b[0]


def get_corner_point(corner, binary_threshold, hough_threshold, min_length, max_gap):

    image_red_channel = corner[:, :, 2]

    _, image_binary = cv2.threshold(image_red_channel, binary_threshold, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(image_binary, 100, 200)

    lines = cv2.HoughLinesP(canny, 1, np.pi/180, hough_threshold, minLineLength=min_length, maxLineGap=max_gap)

    Ax, Ay, Bx, By = lines[0][0]
    m1 = (By - Ay) / (Bx - Ax)
    b1 = Ay - (m1 * Ax)

    Cx, Cy, Dx, Dy = lines[1][0]
    m2 = (Dy - Cy) / (Dx - Cx)
    b2 = Cy - (m2 * Cx)

    x = (b2 - b1) / (m1 - m2)
    y = m1*x + b1

    # corner[np.round(y).astype(int)][np.round(x).astype(int)] = (255, 255, 255)
    # cv2.imshow("hough", corner)

    # gray_mask = cv2.cvtColor(corners_mask, cv2.COLOR_BGR2GRAY)
    # kp = sift.detect(undistorted[:, :, 2], gray_mask)
    # corner = cv2.drawKeypoints(undistorted[:, :, 2], kp, corner)
    # cv2.imshow('sift_keypoints.jpg', corner)

    return np.round(x).astype(int), np.round(y).astype(int)


size = 60
bottom = 310  # 280 - 330
top = 220  # 190 - 250

top_left = 190
top_right = 450

bottom_left = 60
bottom_right = 610

Ax, Ay = get_corner_point(undistorted[bottom-size:bottom + size, bottom_left-size:bottom_left+size],
                          binary_threshold=5, hough_threshold=30, min_length=40, max_gap=10)
Bx, By = get_corner_point(undistorted[top-size:top+size, top_left-size:top_left+size],
                          binary_threshold=5, hough_threshold=40, min_length=30, max_gap=10)
Cx, Cy = get_corner_point(undistorted[top-size:top+size, top_right-size:top_right+size],
                          binary_threshold=30, hough_threshold=30, min_length=30, max_gap=10)
Dx, Dy = get_corner_point(undistorted[bottom-size:bottom+size, bottom_right-size:bottom_right+size],
                          binary_threshold=30, hough_threshold=40, min_length=30, max_gap=10)


Ay = np.round(Ay).astype(int) + bottom - size
Ax = np.round(Ax).astype(int) + bottom_left - size
By = np.round(By).astype(int) + top - size
Bx = np.round(Bx).astype(int) + top_left - size
Cy = np.round(Cy).astype(int) + top - size
Cx = np.round(Cx).astype(int) + top_right - size
Dy = np.round(Dy).astype(int) + bottom - size
Dx = np.round(Dx).astype(int) + bottom_right - size

src_points = np.array([[Ax, Ay], [Bx, By], [Cx, Cy], [Dx, Dy]])

initial_point = 0
width = 640
ratio = 10 / 25
length = np.round(width * ratio).astype(int)
# length = 480

dst_points = np.array([[initial_point, initial_point + length - 1],
                       [initial_point, initial_point],
                       [initial_point + width - 1, initial_point],
                       [initial_point + width - 1, initial_point + length - 1]])

h, _ = cv2.findHomography(src_points, dst_points)

gradient = np.zeros([480, 640, 3], dtype=np.float32)
for i in range(len(gradient)):
    for j in range(len(gradient[i])):
        gradient[i][j] = (1, i/479, j/639)

# cv2.imshow("gradient", gradient)
gradient_undistorted = camera.un_distort(gradient)
gradient_undistorted = cv2.rotate(gradient_undistorted, cv2.ROTATE_180)

gradient_reprojected = cv2.warpPerspective(gradient_undistorted, h, (gradient_undistorted.shape[1], gradient_undistorted.shape[0]))

# cv2.imshow("gradient_reprojected", gradient_reprojected)
# gradient_reprojected[:, :, 1] *= 479
# gradient_reprojected[:, :, 2] *= 639
# gradient_reprojected = np.round(gradient_reprojected).astype(int)


lut = np.zeros([480, 640, 3], dtype=np.float32)
for x in range(len(gradient_reprojected)):
    for y in range(len(gradient_reprojected[x])):
        pix = gradient_reprojected[x][y].copy()

        pix[1] *= 479
        pix[2] *= 639
        pix = np.round(pix).astype(int)

        # if gradient[x][y][1] == np.float32(0) and gradient[x][y][2] == np.float32(0):
        #     print()

        lut[pix[1]][[pix[2]]] = gradient[x][y]

kernel = np.ones((3, 3), np.float32)
lut = cv2.morphologyEx(lut, cv2.MORPH_CLOSE, kernel)

# cv2.imshow("lut", lut)

print(lut[240][320])

new_img3 = np.zeros([480, 640, 3], dtype=np.float32)
# for x in range(len(lut)):
#     for y in range(len(lut[x])):

# cv2.imshow("new_img3", new_img3)

# new_lut = np.array((lut*255), dtype=np.uint8)
# result = cv2.addWeighted(img, 0.7, new_lut, 0.3, 0.0)
# cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
