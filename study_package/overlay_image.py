import cv2
import numpy as np

base_image = cv2.imread("reprojected.png")

normalized_image = np.zeros([480, 640, 3], dtype=np.float32)
for i in range(len(normalized_image)):
    for j in range(len(normalized_image[i])):
        normalized_image[i][j] = (1, i / (480 - 1), j / (640 - 1))

norm_int = 255 * normalized_image
norm_int = np.array(norm_int, dtype="uint8")

result = cv2.addWeighted(base_image, 0.5, norm_int, 0.7, 0.0)
cv2.imwrite("reproj_overlay.png", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
