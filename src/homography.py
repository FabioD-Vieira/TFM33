import cv2
import numpy as np


class Homography:

    def __init__(self, camera_resolution):
        self.__width, self.__height = camera_resolution
        self.__h = None

        # base image pool corners
        self.__base_corners = [(199, 192), (195, 454), (324, 37), (320, 638)]

        self.__sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=10)
        self.__bf = cv2.BFMatcher()

        self.__RED = 2

    def __find_difference(self, corner_base_img, corner_img):

        # detect features in each image and compare them
        kp1, des1 = self.__sift.detectAndCompute(corner_base_img[:, :, 2], None)
        kp2, des2 = self.__sift.detectAndCompute(corner_img[:, :, 2], None)

        matches = self.__bf.knnMatch(des1, des2, k=2)

        ratio_dist = 0.5
        vector = np.array([0, 0], dtype='float')
        number_of_features = 0
        for m, n in matches:
            # Apply ratio test to remove useless features
            if m.distance < ratio_dist * n.distance:
                (x1, y1) = kp1[m.queryIdx].pt
                (x2, y2) = kp2[m.trainIdx].pt

                # calculate difference between each feature match and sum it in a single vector
                vector += np.array([x2 - x1, y2 - y1])
                number_of_features += 1

        # Represent matches in a single image
        # good = [[m] for m, n in matches if m.distance < ratio_dist * n.distance]
        # img3 = cv2.drawMatchesKnn(corner_base_img, kp1, corner_img, kp2, good, None,
        #                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #
        # cv2.imshow("match", img3)

        # vector mean
        vector = vector / number_of_features
        vector_x, vector_y = np.round(vector).astype(int)

        return vector_y, vector_x

    def __calculate_source_points(self, base_image, image):

        # Calculates the difference between two images for each corner section
        # and adds the difference to each know corner point

        # Top left
        vector_y, vector_x = self.__find_difference(base_image[150:250, 150:250], image[150:250, 150:250])
        i, j = self.__base_corners[0]
        ay, ax = i + vector_y, j + vector_x

        # Top right
        vector_y, vector_x = self.__find_difference(base_image[150:250, 400:500], image[150:250, 400:500])
        i, j = self.__base_corners[1]
        by, bx = i + vector_y, j + vector_x

        # Bottom left
        vector_y, vector_x = self.__find_difference(base_image[275:375, 0:100], image[275:375, 0:100])
        i, j = self.__base_corners[2]
        cy, cx = i + vector_y, j + vector_x

        # Bottom right
        vector_y, vector_x = self.__find_difference(base_image[270:370, 540:640], image[270:370, 540:640])
        i, j = self.__base_corners[3]
        dy, dx = i + vector_y, j + vector_x

        return np.array([[ax, ay], [bx, by], [cx, cy], [dx, dy]])

    def calculate_homography_matrix(self, base_image, image):

        src = self.__calculate_source_points(base_image, image)
        dst = np.array([[0, 0], [self.__width - 1, 0],
                        [0, self.__height - 1], [self.__width - 1, self.__height - 1]])

        self.__h, _ = cv2.findHomography(src, dst)

    def apply_homography(self, image):
        return cv2.warpPerspective(image, self.__h, (image.shape[1], image.shape[0]))
