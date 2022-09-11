import cv2

from src.image_processing.setup.camera import Camera
from src.image_processing.setup.homography import Homography
from src.image_processing.setup.lut import LUT


class Setup:

    def __init__(self, resolution, balance):
        self.__camera = Camera(resolution, balance)
        self.homography = Homography(resolution)
        self.__lut = LUT(self.__camera, self.homography)

    def calibrate_camera(self, calibration_images):
        self.__camera.calibrate(calibration_images)

    def calculate_homography_matrix(self, base_image, img):
        base_undistorted = self.__camera.un_distort(base_image)
        undistorted = self.__camera.un_distort(img)

        base_rotated = cv2.rotate(base_undistorted, cv2.ROTATE_180)
        rotated = cv2.rotate(undistorted, cv2.ROTATE_180)

        self.homography.calculate_homography_matrix(base_rotated, rotated)

    def generate_lut(self):
        return self.__lut.generate_lut()

    # def no_lut_process(self, image):
    #     undistorted = self.__camera.un_distort(image)
    #     rotated = cv2.rotate(undistorted, cv2.ROTATE_180)
    #
    #     return self.homography.apply_homography(rotated)
