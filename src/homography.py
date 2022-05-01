import cv2
import numpy as np


def get_corner_point(corner, binary_threshold, hough_threshold, min_length, max_gap):

    image_red_channel = corner[:, :, 2]

    _, image_binary = cv2.threshold(image_red_channel, binary_threshold, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(image_binary, 100, 200)

    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, hough_threshold, minLineLength=min_length, maxLineGap=max_gap)

    # Find lines intersection point
    ax, ay, bx, by = lines[0][0]
    m1 = (by - ay) / (bx - ax)
    b1 = ay - (m1 * ax)

    cx, cy, dx, dy = lines[1][0]
    m2 = (dy - cy) / (dx - cx)
    b2 = cy - (m2 * cx)

    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return np.round(x).astype(int), np.round(y).astype(int)


class Homography:

    def __init__(self, width, height, pool_dim):
        self.__h = None

        self.__width = width
        # self.__length = height
        ratio = pool_dim[1] / pool_dim[0]
        self.__length = np.round(self.__width * ratio).astype(int)

        self.__size = 60
        self.__bottom = 310  # 280 - 330
        self.__top = 220  # 190 - 250

        self.__top_left = 190
        self.__top_right = 450

        self.__bottom_left = 60
        self.__bottom_right = 610

    def __calculate_source_points(self, image):

        # Bottom left
        ax, ay = get_corner_point(image[self.__bottom-self.__size:self.__bottom+self.__size,
                                  self.__bottom_left-self.__size:self.__bottom_left+self.__size],
                                  binary_threshold=5, hough_threshold=30, min_length=40, max_gap=10)

        ay = np.round(ay).astype(int) + self.__bottom - self.__size
        ax = np.round(ax).astype(int) + self.__bottom_left - self.__size

        # Top left
        bx, by = get_corner_point(image[self.__top-self.__size:self.__top+self.__size,
                                  self.__top_left-self.__size:self.__top_left+self.__size],
                                  binary_threshold=5, hough_threshold=40, min_length=30, max_gap=10)

        by = np.round(by).astype(int) + self.__top - self.__size
        bx = np.round(bx).astype(int) + self.__top_left - self.__size

        # Top right
        cx, cy = get_corner_point(image[self.__top-self.__size:self.__top+self.__size,
                                  self.__top_right-self.__size:self.__top_right+self.__size],
                                  binary_threshold=30, hough_threshold=30, min_length=30, max_gap=10)

        cy = np.round(cy).astype(int) + self.__top - self.__size
        cx = np.round(cx).astype(int) + self.__top_right - self.__size

        # Bottom right
        dx, dy = get_corner_point(image[self.__bottom-self.__size:self.__bottom+self.__size,
                                  self.__bottom_right-self.__size:self.__bottom_right+self.__size],
                                  binary_threshold=30, hough_threshold=40, min_length=30, max_gap=10)

        dy = np.round(dy).astype(int) + self.__bottom - self.__size
        dx = np.round(dx).astype(int) + self.__bottom_right - self.__size

        return np.array([[ax, ay], [bx, by], [cx, cy], [dx, dy]])

    def calculate_homography_matrix(self, image):

        src = self.__calculate_source_points(image)
        dst = np.array([[0, self.__length - 1], [0, 0], [self.__width - 1, 0], [self.__width - 1, self.__length - 1]])

        self.__h, _ = cv2.findHomography(src, dst)

    def apply_homography(self, image):
        return cv2.warpPerspective(image, self.__h, (image.shape[1], image.shape[0]))
