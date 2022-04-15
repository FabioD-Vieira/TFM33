import cv2
import numpy as np

from src import pool_utils


class PoolProcessing:

    def __init__(self):
        self.__dim = (640, 480)
        self.__matrix = None
        self.__coefficients = None
        self.__h = None

        # TODO Study
        self.__dim2 = None
        self.__dim3 = None

        if not self.__dim2:
            self.__dim2 = self.__dim
        if not self.__dim3:
            self.__dim3 = self.__dim

        # assuming pool size of 25 per 10 meters
        self.__pool_width = 10
        self.__pool_length = 25
        ratio = self.__pool_width / self.__pool_length

        self.__length = 450  # pixels for pool length
        self.__width = self.__length * ratio
        self.__initial_point = 80  # top left / margin

    def calibrate(self, calibration_images):

        pattern = (6, 9)

        real_world_points = []
        image_points = []

        points = np.zeros((1, pattern[0] * pattern[1], 3), np.float32)
        points[0, :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

        corner_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        for image_name in calibration_images:
            img = cv2.imread(image_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, pattern, flags=corner_flags)

            if ret:
                real_world_points.append(points)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), corner_criteria)
                image_points.append(corners)

                # Draw corners
                # cv2.drawChessboardCorners(img, pattern, corners, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)

        self.__matrix = np.zeros((3, 3))
        self.__coefficients = np.zeros((4, 1))

        calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        calib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        cv2.fisheye.calibrate(real_world_points, image_points, self.__dim, self.__matrix, self.__coefficients,
                              flags=calib_flags,  criteria=calib_criteria)

        # print("matrix: " + str(self.__matrix.tolist()))
        # print("coefficients:" + str(self.__coefficients.tolist()))

    def un_distort(self, image, balance):

        new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.__matrix, self.__coefficients,
                                                                            self.__dim2, np.eye(3), balance=balance)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.__matrix, self.__coefficients, np.eye(3),
                                                         new_matrix, self.__dim3, cv2.CV_16SC2)

        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def calculate_homography_matrix(self):

        # Pool: 2      3
        #
        #       1      4
        # Obtained by clicking in the image (pool corners)
        src_points = np.array([[86, 78], [263, 129], [435, 264], [566, 469]])

        dst_points = np.array([[self.__initial_point, self.__initial_point + self.__width],
                               [self.__initial_point, self.__initial_point],
                               [self.__initial_point + self.__length, self.__initial_point],
                               [self.__initial_point + self.__length, self.__initial_point + self.__width]])

        self.__h, _ = cv2.findHomography(src_points, dst_points)

    def apply_homography(self, image):
        return cv2.warpPerspective(image, self.__h, (image.shape[1], image.shape[0]))

    def get_vessel_location(self, image):
        location_in_image = pool_utils.get_location_in_image(image)

        location_image_no_margin = location_in_image - self.__initial_point
        x = location_image_no_margin[1] * self.__pool_length / self.__length
        y = location_image_no_margin[0] * self.__pool_width / self.__width

        return x, y
