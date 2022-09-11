import glob
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.image_processing.setup import Camera
from src.image_processing.setup.setup import Setup

system = Setup()
camera = Camera()

images = glob.glob('../images/calibration/*.jpg')
camera.calibrate(images)  # Called only once to calibrate

img = cv2.imread("../../images/pool/img06.jpg")
undistorted = camera.un_distort(img, balance=0.9)
cv2.imshow("und", undistorted)

top_vector = math.sqrt((435 - 263)**2 + (264 - 129)**2) / 25
bottom_vector = math.sqrt((566 - 86)**2 + (469 - 78)**2) / 25

top_prob = top_vector / bottom_vector
bottom_prob = 1

ratio = (bottom_prob - top_prob) / 479

prob = np.zeros([480, 640], dtype=np.float32)
for i in range(480):
    val = i * ratio
    for j in range(640):
        prob[i][j] = top_prob + val

# plt.imshow(prob, cmap='hot', interpolation='nearest')
# plt.show()

fig, axis = plt.subplots()
heatmap = axis.pcolor(prob)

cbar = fig.colorbar(heatmap, ticks=[prob[0][0], prob[239][0], prob[-1][0]])
cbar.ax.set_yticklabels([np.round(prob[0][0]*bottom_vector, 2),
                         np.round(prob[239][0]*bottom_vector, 2),
                         str(np.round(prob[-1][0]*bottom_vector, 2)) + ' pixels/m'])
cbar.ax.invert_yaxis()

# axis.set_yticks(np.arange(prob.shape[0])+0.5, minor=False)
# axis.set_xticks(np.arange(prob.shape[1])+0.5, minor=False)

axis.invert_yaxis()

# row_labels = ["Lundi",
#               "Mardi",
#               "Mercredi",
#               "Jeudi",
#               "Vendredi",
#               "Samedi",
#               "Dimanche"]

# axis.set_yticklabels(row_labels, minor=False)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
