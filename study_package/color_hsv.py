import cv2
import numpy as np
from matplotlib import pyplot as plt

# plt.figure(figsize=(10,5))
plt.xlabel("Saturation")
plt.ylabel("Hue")

img = np.zeros([180, 255, 3], dtype='uint8')

for i in range(len(img)):
    for j in range(len(img[i])):
        img[i][j] = (i, j, 255)

img_rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
# plt.plot(img_rgb)
# cv2.imwrite("colors.png", img_rgb)
plt.imshow(img_rgb)
plt.show()
# cv2.imshow("ff", img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
