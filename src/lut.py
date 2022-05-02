import cv2
import numpy as np


class LUT:
    def __init__(self, camera, homography):
        self.__camera = camera
        self.__cam_width, self.__cam_height = self.__camera.dim()

        self.__homography = homography

        self.__lut = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.uint)

    def generate_lut(self, img):

        undistorted = self.__camera.un_distort(img)
        rotated = cv2.rotate(undistorted, cv2.ROTATE_180)

        self.__homography.calculate_homography_matrix(rotated)

        gradient_image = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.float32)
        for i in range(len(gradient_image)):
            for j in range(len(gradient_image[i])):
                gradient_image[i][j] = (1, i / (self.__cam_height - 1), j / (self.__cam_width - 1))

        gradient_undistorted = self.__camera.un_distort(gradient_image)
        gradient_undistorted = cv2.rotate(gradient_undistorted, cv2.ROTATE_180)

        gradient_reprojected = self.__homography.apply_homography(gradient_undistorted)
        gradient_reprojected[:, :, 1] *= (self.__cam_height - 1)
        gradient_reprojected[:, :, 2] *= (self.__cam_width - 1)

        gradient_reprojected = np.round(gradient_reprojected).astype(int)
        for x in range(len(gradient_reprojected)):
            for y in range(len(gradient_reprojected[x])):
                self.__lut[x][y] = gradient_reprojected[x][y]

    def apply_lut(self, image):
        new_image = np.array(image)

        for x in range(len(self.__lut)):
            for y in range(len(self.__lut[x])):
                _, row, column = self.__lut[x][y]
                new_image[x][y] = image[row][column]

        return new_image
