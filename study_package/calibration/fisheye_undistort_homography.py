import cv2
import numpy as np

image_path = "../../images/pool/img06.jpg"
img = cv2.imread(image_path)

DIM = (640, 480)
K = np.array(
    [[275.39128969212095, 0.0, 320.5852839689208], [0.0, 275.6790638659298, 222.30555908161557], [0.0, 0.0, 1.0]])
D = np.array([[-0.059424647706354174], [0.04070772905851004], [-0.05191779330361861], [0.017710952492668797]])

dim1 = img.shape[:2][::-1]
dim2 = None
dim3 = None

balance = 0.9

if not dim2:
    dim2 = dim1
if not dim3:
    dim3 = dim1

scaled_K = K * dim1[0] / DIM[0]  # The values of K are to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

print(scaled_K)

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("Undistorted", undistorted)


def print_coordinates(event, x_coord, y_coord, _, _1):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x_coord, y_coord)


cv2.setMouseCallback("Undistorted", print_coordinates)

src_points = np.array([[86, 78], [263, 129], [435, 262], [566, 469]])

length = 450
ratio = 10 / 25
width = length * ratio

initial_point = 80

dst_points = np.array([[initial_point, initial_point + width], [initial_point, initial_point],
                       [initial_point + length, initial_point],
                       [initial_point + length, initial_point + width]])

h, _ = cv2.findHomography(src_points, dst_points)

new_image = cv2.warpPerspective(undistorted, h, (undistorted.shape[1], undistorted.shape[0]))
cv2.imshow("homography", new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
