import cv2
import numpy as np


class Camera:

    def __init__(self, resolution, balance):
        self.__dim = resolution
        self.__balance = balance

        self.__coefficients = None
        self.__matrix = None

        # TODO Study
        self.__dim2 = None
        self.__dim3 = None

        if not self.__dim2:
            self.__dim2 = self.__dim
        if not self.__dim3:
            self.__dim3 = self.__dim

    def dim(self):
        return self.__dim

    def calibrate(self, calibration_images):

        pattern = (6, 9)

        real_world_points = []
        image_points = []

        points = np.zeros((1, pattern[0] * pattern[1], 3), np.float32)
        points[0, :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

        corner_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        for image in calibration_images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, pattern, flags=corner_flags)

            if ret:
                real_world_points.append(points)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), corner_criteria)
                image_points.append(corners)

                # Draw corners
                # cv2.drawChessboardCorners(img, pattern, corners, ret)
                # cv2.imshow('img', image)
                # cv2.waitKey(500)

        self.__matrix = np.zeros((3, 3))
        self.__coefficients = np.zeros((4, 1))

        calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        calib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        cv2.fisheye.calibrate(real_world_points, image_points, self.__dim, self.__matrix, self.__coefficients,
                              flags=calib_flags,  criteria=calib_criteria)

        # print("matrix: " + str(self.__matrix.tolist()))
        # print("coefficients:" + str(self.__coefficients.tolist()))

    def un_distort(self, image):

        new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.__matrix, self.__coefficients,
                                                                            self.__dim2, np.eye(3),
                                                                            balance=self.__balance)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.__matrix, self.__coefficients, np.eye(3),
                                                         new_matrix, self.__dim3, cv2.CV_16SC2)

        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
