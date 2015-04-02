####
# Figure S7
# needs:
# - data/res.npz produced by run.py
####

import sys
sys.path.append('..')
from lib.mppaper import *

import matplotlib.cm as cm
from matplotlib.patches import Circle

#### load data ####
npz = np.load('data/res.npz')
P = npz['P']
x = npz['x']
N = len(npz['x'])
self = npz['self']
import run
selfexwidth = run.self_exclusion_width
sigma = run.sigma

#### plot ####
fig = plt.figure(figsize=(figsize[0], figsize[0]*0.85))
ax = fig.add_subplot(111)
for s in self:
    for xshift in [-1, 0, 1]:
        for yshift in [-1, 0, 1]:
            c = Circle(xy=(s + np.array([xshift, yshift]))/sigma,
                       # exclude outer layer of one pixel
                       radius=selfexwidth - np.mean(np.diff(x))/sigma,
                       fc=cmediumprime, ec='none')
            ax.add_patch(c)
im = ax.imshow(P.reshape(N, N), cmap=cm.gray_r,
               extent=(min(x)/sigma, max(x)/sigma, min(x)/sigma, max(x)/sigma))
cbar = fig.colorbar(im, ticks=(0.0,), fraction=0.05)
# workaround for pdf/svg export for more smoothness
# see colorbar documentation
cbar.solids.set_edgecolor("face")
cbar.ax.set_yticklabels(['0'])
cbar.ax.set_ylabel('$P^\star_r$')
ax.set_xlabel('$r_1 \; / \; \sigma$')
ax.set_ylabel('$r_2 \; / \;  \sigma$')

#### Finalize figure ####
fig.tight_layout(pad=tight_layout_pad)
fig.savefig('figS7.svg')
plt.show()
