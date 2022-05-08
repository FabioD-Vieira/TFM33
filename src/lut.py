import cv2
import numpy as np


class LUT:

    def __init__(self, camera, homography, pool_dim):
        self.__camera = camera
        self.__cam_width, self.__cam_height = self.__camera.dim()

        self.__homography = homography

        self.__lut = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.float32)

    def generate_lut(self, base_image, img):

        base_undistorted = self.__camera.un_distort(base_image)
        base_rotated = cv2.rotate(base_undistorted, cv2.ROTATE_180)

        undistorted = self.__camera.un_distort(img)
        rotated = cv2.rotate(undistorted, cv2.ROTATE_180)

        self.__homography.calculate_homography_matrix(base_rotated, rotated)

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

        # LUT to reconstruct final image perfectly
        diff_lut = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.uint)

        lut = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.float32)
        for x in range(len(gradient_reprojected)):
            for y in range(len(gradient_reprojected[x])):
                pix = gradient_reprojected[x][y]
                lut[pix[1]][[pix[2]]] = gradient_image[x][y]

                diff_lut[x][y] = gradient_reprojected[x][y]

        kernel = np.ones((3, 3), np.float32)
        self.__lut = cv2.morphologyEx(lut, cv2.MORPH_CLOSE, kernel)

        # Reconstruct image badly with normal LUT
        # reconstruct_image_badly(img, self.__lut)

        # Reconstruct image perfectly with different LUT
        # reconstruct_image_perfectly(img, diff_lut)

    def apply_lut(self, i, j):

        pix = self.__lut[i][j]
        return pix[2], pix[1]


def __reconstruct_image_badly(image, lut):
    new_image = np.zeros([480, 640, 3], dtype=np.uint8)

    for x in range(len(image)):
        for y in range(len(image[x])):
            pix = lut[x][y]
            k, p = pix[1] * 479, pix[2] * 639
            new_image[int(np.round(k))][int(np.round(p))] = image[x][y]

    cv2.imshow("bad reconstruction", new_image)


def reconstruct_image_perfectly(image, lut):
    new_image = np.zeros([480, 640, 3], dtype=np.uint8)

    for x in range(len(lut)):
        for y in range(len(lut[x])):
            _, row, column = lut[x][y]
            new_image[x][y] = image[row][column]

    cv2.imshow("perfect reconstruction", new_image)
