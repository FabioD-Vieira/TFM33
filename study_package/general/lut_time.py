
from matplotlib import pyplot as plt

time_lut = [0.1151, 0.1157, 0.1173, 0.1219, 0.1238]
time_no_lut = [47.3949, 47.4208, 47.4995, 47.6820, 47.4795]
res = ["VGA", "HD", "Full HD", "4K", "8K"]

f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

ax.plot(res, time_no_lut, marker='o', color='orange')
ax2.plot(res, time_lut, marker='o')

ax.set_ylim(45, 48)
ax2.set_ylim(0.11, 0.15)

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.xaxis.tick_bottom()
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax.xaxis.tick_top()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

plt.xlabel("Resolution")
ax.set_ylabel("Time per 1000 frames")
ax.yaxis.set_label_coords(-0.1, 0)

plt.show()

