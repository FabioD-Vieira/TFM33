import cv2
import numpy as np


class Camera:

    def __init__(self, resolution, balance):
        self.__dim = resolution
        self.__balance = balance

        self.__map1, self.__map2 = None, None

    def dim(self):
        return self.__dim

    def calibrate(self, calibration_images):

        pattern = (6, 9)

        real_world_points = []
        image_points = []

        # Create points to be used as real world references
        points = np.zeros((1, pattern[0] * pattern[1], 3), np.float32)
        points[0, :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

        for image in calibration_images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # find chessboard pattern in each calibration image
            ret, corners = cv2.findChessboardCorners(gray, pattern)

            # if corners were found, add reference points to real world references
            if ret:
                real_world_points.append(points)
                image_points.append(corners)

                # Draw corners
                # cv2.drawChessboardCorners(image, pattern, corners, ret)
                # cv2.imshow('img', image)
                # cv2.waitKey(500)

        # Initialize variables
        matrix = np.zeros((3, 3))
        coefficients = np.zeros((4, 1))

        calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC

        # Update matrix and coefficient variables with calibration
        cv2.fisheye.calibrate(real_world_points, image_points, self.__dim, matrix, coefficients, flags=calib_flags)

        # functions called inside fisheye.undistort but are heavy, so it's best to split them
        new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(matrix, coefficients, self.__dim, np.eye(3),
                                                                            balance=self.__balance)

        self.__map1, self.__map2 = cv2.fisheye.initUndistortRectifyMap(matrix, coefficients, np.eye(3), new_matrix,
                                                                       self.__dim, cv2.CV_16SC2)

    def un_distort(self, image):
        return cv2.remap(image, self.__map1, self.__map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
