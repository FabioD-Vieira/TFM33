import glob

import numpy as np
import cv2

chess = (6, 9)
res = (640, 480)

objp = np.zeros((chess[0] * chess[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chess[0], 0:chess[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

images = glob.glob('../../images/calibration/*.jpg')

number_of_images = 0
print(len(images))

for name in images:
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chess, None)

    if ret:
        number_of_images += 1

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chess, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

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
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))
