from src.image_processing.setup.homography import Homography
from src.image_processing.setup.lut import LUT


class Setup:

    def __init__(self, camera, pool_dim):
        self.__camera = camera
        self.__cam_width, self.__cam_height = self.__camera.dim()

        self.__pool_dim = pool_dim

        self.homography = Homography(self.__camera.dim())
        self.__lut = LUT(self.__camera, self.homography)

    def calibrate_camera(self, calibration_images):
        self.__camera.calibrate(calibration_images)

    def generate_lut(self, base_image, img):
        return self.__lut.generate_lut(base_image, img)

    # def process(self, img):
    #     undistorted = self.__camera.un_distort(img)
    #     rotated = cv2.rotate(undistorted, cv2.ROTATE_180)
    #
    #     cv2.imshow("rotated", rotated)
    #     reprojected = self.homography.apply_homography(rotated)
    #     cv2.imshow("No LUT", reprojected)