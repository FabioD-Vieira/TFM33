import cv2
import numpy as np


class LUT:

    def __init__(self, camera, homography):
        self.__camera = camera
        self.__cam_width, self.__cam_height = self.__camera.dim()

        self.__homography = homography

        self.__lut = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.float32)

    def generate_lut(self, base_image, img):

        # Compute homography matrix using a known image (base_image)
        # and a new image obtain from the camera
        base_undistorted = self.__camera.un_distort(base_image)
        base_rotated = cv2.rotate(base_undistorted, cv2.ROTATE_180)

        undistorted = self.__camera.un_distort(img)
        rotated = cv2.rotate(undistorted, cv2.ROTATE_180)

        self.__homography.calculate_homography_matrix(base_rotated, rotated)

        # Created normalized image with each pixel value corresponding to a position: (1, i/479, j/639)
        normalized_image = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.float32)
        for i in range(len(normalized_image)):
            for j in range(len(normalized_image[i])):
                normalized_image[i][j] = (1, i / (self.__cam_height - 1), j / (self.__cam_width - 1))

        # Apply the same transformations
        normalized_undistorted = self.__camera.un_distort(normalized_image)
        normalized_rotated = cv2.rotate(normalized_undistorted, cv2.ROTATE_180)
        normalized_reprojected = self.__homography.apply_homography(normalized_rotated)

        # Transforms pixel values in positions
        normalized_reprojected[:, :, 1] *= (self.__cam_height - 1)
        normalized_reprojected[:, :, 2] *= (self.__cam_width - 1)
        normalized_reprojected = np.round(normalized_reprojected).astype(int)

        # LUT to reconstruct final image perfectly
        # diff_lut = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.uint)

        # create LUT by associating each position to a value (position) of the normalized image
        lut = np.zeros([self.__cam_height, self.__cam_width, 3], dtype=np.float32)
        for x in range(len(normalized_reprojected)):
            for y in range(len(normalized_reprojected[x])):
                pix = normalized_reprojected[x][y]
                lut[pix[1]][[pix[2]]] = normalized_image[x][y]

                # diff_lut[x][y] = normalized_reprojected[x][y]

        kernel = np.ones((3, 3), np.float32)
        self.__lut = cv2.morphologyEx(lut, cv2.MORPH_CLOSE, kernel)

        # Reconstruct image badly with normal LUT
        # reconstruct_image_badly(img, self.__lut)

        # Reconstruct image perfectly with different LUT
        # reconstruct_image_perfectly(img, diff_lut)

    def apply(self, i, j):
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
