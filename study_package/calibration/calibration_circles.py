import glob

import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
chess = (6, 4)

res = (640, 480)

objp = np.zeros((chess[0] * chess[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chess[0], 0:chess[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('../../images/calibration_circles/*.jpg')

number_of_images = 0
print(len(images))

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 1
params.maxThreshold = 255

# params.filterByConvexity = True
# params.minConvexity = 0.4

# params.filterByArea = True
# params.minArea = 50
# params.maxArea = 300

# params.filterByInertia = True
# params.minInertiaRatio = 0.5

# params.filterByCircularity = True
# params.minCircularity = 0.8

# params.minDistBetweenBlobs = 7

detector = cv2.SimpleBlobDetector_create(params)

for name in images:
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findCirclesGrid(gray, chess, None, flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING),
                                       blobDetector=detector)

    if ret:
        number_of_images += 1

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

cv2.destroyAllWindows()
print(number_of_images)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, res, None, None)

print("###############")
print(ret)
print("###############")
print(mtx)
print("###############")
print(dist)
print("###############")
print(rvecs)
print("###############")
print(tvecs)
print("###############")

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
