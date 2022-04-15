import cv2 as cv
import numpy as np

image_path = "../../../images/pool/img06.jpg"
image = cv.imread(image_path)

cv.imshow("Image", image)

#######
# mtx = np.array([[284.09199125, 0., 333.58468528],
#                 [0., 286.57231784, 228.78259243],
#                 [0., 0., 1.]])
#
# dist = np.array([[-0.27082553, 0.05670376, -0.00057874, -0.00264447, -0.00464092]])

#######
# mtx = np.array([[273.34618059, 0., 309.49630399],
#                 [0., 273.95108651, 217.96747154],
#                 [0., 0., 1.]])
#
# dist = np.array([[-0.26575084, 0.05453926, 0.00125302, 0.00155206, -0.00410494]])
#

#######
# mtx = np.array([[270.63540133, 0., 327.88446918],
#                 [0., 270.85324407, 222.86103904],
#                 [0., 0., 1., ]])
#
# dist = np.array([[-0.27714108, 0.06291579, -0.00124666, -0.00055931, -0.00553095]])

########
# mtx = np.array([[271.57352704, 0., 316.43053844],
#                 [0., 271.96067817, 219.37458422],
#                 [0., 0., 1.]])
#
# dist = np.array([[-2.64809487e-01, 5.46564118e-02, 6.35828879e-04, 1.08756183e-04,
#                   -4.22546892e-03]])

# Circles
mtx = np.array([[271.56002755, 0., 326.48381859],
                [0., 272.40542068, 217.92025054],
                [0., 0., 1.]])

dist = np.array([[-2.47421285e-01, 4.61916330e-02, 1.73875635e-04, -7.26108802e-04,
                  -3.28833132e-03]])

h, w = image.shape[:2]
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.4, (w, h))

undistorted = cv.undistort(image, mtx, dist, None, new_camera_matrix)
cv.imshow("Undistorted", undistorted)

# x, y, w, h = roi
# undistorted_crop = undistorted[y:y + h, x:x + w]
# cv.imshow("Undistorted cropped", undistorted_crop)


cv.waitKey(0)
cv.destroyAllWindows()
