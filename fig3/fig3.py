####
# Figure 3
# needs:
# - data/1d.npz produced by run1d.py
# - data/2d.npz produced by run2d.py
####

import sys
sys.path.append('..')
from lib.mppaper import *
import lib.mpsetup as mpsetup
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

thisfigsize = figsize
thisfigsize[1] *= 0.75
fig = plt.figure(figsize=thisfigsize)
grid = gridspec.GridSpec(1, 2, left=0.06, right=0.97, top=0.91, bottom=0.2,
                         wspace=0.4, hspace=0.35)
labeled_axes = []

#### subfigure A ####
## import data
import run1d
sigma = run1d.sigma
npz = np.load('data/1d.npz')
Q = npz['Q']
x = npz['x']
P = npz['P']
Ptilde = npz['Ptilde']

## plot
ax = plt.Subplot(fig, grid[0, 0])
fig.add_subplot(ax)
labeled_axes.append(ax)
ax.plot(x/sigma, Q, label=r'$Q_a$', **linedotstyle)
ax.plot(x/sigma, P, label=r'$P^\star_r$', **linedotstyle)
ax.plot(x/sigma, Ptilde, linestyle='-', label=r'$\tilde P^\star_a$')
ax.set_yticks([])
ax.set_xlabel('$x \; / \;  \sigma$')
ax.set_ylabel('probability')
ax.xaxis.labelpad = axis_labelpad
ax.yaxis.labelpad = axis_labelpad
ax.legend(frameon=False, ncol=3, columnspacing=0.5, handletextpad=0.2,
          loc='upper right', bbox_to_anchor=(1.05, 1.18))
mpsetup.despine(ax)

#### subfigure B ####
## import data
import run2d
sigma = run2d.sigma
N = run2d.N
npz = np.load('data/2d.npz')
Q = npz['Q']
popt = npz['P']

## plot
ax = plt.Subplot(fig, grid[0, 1])
fig.add_subplot(ax)
labeled_axes.append(ax)
vmax = np.amax(popt) * N**2
im = ax.imshow(popt.reshape(N, N)*N**2, extent=(0, 1.0/sigma, 0, 1.0/sigma),
               # same as nearest, but better for svg export
               interpolation = 'none',
               vmax = vmax, cmap = cm.gray_r)
cbar = fig.colorbar(im, ticks=(0.0,))
# workaround for pdf/svg export for more smoothness
# see matplotlib colorbar documentation
cbar.solids.set_edgecolor("face")
cbar.ax.set_yticklabels(['0'])
cbar.ax.set_ylabel('$P^\star_r$')
cbar.ax.yaxis.set_label_coords(2.0, 0.5)
ticks = np.arange(0.0, 25.0, 5.0)
ax.set_yticks(ticks)
ax.set_xticks(ticks)
ax.set_xlabel('$r_1 \; / \; \sigma$')
ax.set_ylabel('$r_2 \; / \;  \sigma$')
ax.xaxis.labelpad = axis_labelpad
ax.yaxis.labelpad = axis_labelpad

#### finish figure ####
labeldict = dict(labelstyle=r'{\sf \textbf{%s}}', fontsize='medium',
                 xycoords=('axes fraction'), fontweight = 'bold')
mpsetup.label_axes([labeled_axes[0]], labels='A', xy=(-0.12, 0.95), **labeldict)
mpsetup.label_axes([labeled_axes[1]], labels='B', xy=(-0.3, 0.95), **labeldict)
fig.savefig('fig3.svg')
plt.show()
