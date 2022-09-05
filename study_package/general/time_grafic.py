import math

from matplotlib import pyplot as plt

time_lut = [0.0012, 0.0012, 0.0012, 0.0012]
time_und = [0.1620, 0.6153, 1.2685, 5.0127]
time_repr = [0.2496, 0.5404, 1.1266, 4.4036]
time_calib_homography = [0.4116, 1.1557, 2.3951, 9.4163]
res = ["VGA", "HD", "Full HD", "4K"]

plt.plot(res, [math.log10(i) for i in time_calib_homography], marker='o', color='orange')
plt.plot(res, [math.log10(i) for i in time_und], marker='o', color='red')
plt.plot(res, [math.log10(i) for i in time_repr], marker='o', color='green')
plt.plot(res, [math.log10(i) for i in time_lut], marker='o')

plt.ylabel("t(s)")
plt.xlabel("Resolution")

plt.legend(["Undistortion + Re-projection", "Undistortion", "Re-projection", "Proposed method (LUT)"])

plt.show()
