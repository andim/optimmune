####
# Figure 5
# needs:
# - data/out.npz produced by run.py
####
import sys
sys.path.append('..')
from lib.mppaper import *
import lib.mpsetup as mpsetup

import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#### import data ####
npz = np.load('data/out.npz')
ns_mf = npz['ns_mf']
ts_stoch = npz['ts_stoch']
ts_mf = npz['ts_mf']
costs_mf = npz['costs_mf']
costs_stoch = npz['costs_stoch']
costs_stoch2 = npz['costs_stoch2']
costs_stoch3 = npz['costs_stoch3']
fopt = npz['fopt']
N = len(npz['x'])
Q = npz['Q']
x = npz['x']
N = len(x)

import run
lambda_ = run.lambda_
tstart, tend = min(ts_mf)*lambda_, max(ts_mf)*lambda_

tlinend = 5
numplot = [5, 61, -1]

fig = plt.figure()

#### First part of figure (small times) ####
ax1 = fig.add_subplot(121)
mpsetup.despine(ax1)
costs_norm = (costs_mf - fopt)/fopt
m, = ax1.plot(ts_mf * lambda_, costs_norm)
s1, = ax1.plot(ts_stoch * lambda_, (costs_stoch - fopt)/fopt,
               lw=linewidth*0.5)
s2, = ax1.plot(ts_stoch * lambda_, (costs_stoch2 - fopt)/fopt,
               lw=linewidth*0.5)
s3, = ax1.plot(ts_stoch * lambda_, (costs_stoch3 - fopt)/fopt,
               lw=linewidth*0.5)
ax1.set_ylabel(r'relative cost gap')
ax1.xaxis.labelpad = axis_labelpad
ax1.set_xlim(tstart, tlinend)
ax1.locator_params(axis='y', nbins=5)
ax1.set_xticks(range(5))
ax1.legend([m, (s1, s2, s3)], ['mean field', 'stochastic'],
           handler_map={tuple: mpsetup.OffsetHandlerTuple()},
           handletextpad=0.2, loc='upper center', bbox_to_anchor=(0.5, 0.6),
           fontsize = 'small')

#### Second part of figure (large times) ####
ax2 = fig.add_subplot(122, sharey=ax1, zorder=-1)
ax2.plot(ts_mf * lambda_, costs_norm)
# rasterize these lines to trim down figure size
ax2.plot(ts_stoch * lambda_, (costs_stoch - fopt)/fopt,
         lw=linewidth*0.5, rasterized=True)
ax2.plot(ts_stoch * lambda_, (costs_stoch2 - fopt)/fopt,
         lw=linewidth*0.5, rasterized=True)
ax2.plot(ts_stoch * lambda_, (costs_stoch3 - fopt)/fopt,
         lw=linewidth*0.5, rasterized=True)
ax2.set_xlim(tlinend, tend)
ax2.set_xscale('log')
mpsetup.despine(ax2)
ax2.spines['left'].set_visible(False)
ax2.tick_params(labelleft='off', left='off')

#### Setup main figure ####
# fake xlabel to fool tight layout
ax1.set_xlabel('time')
fig.tight_layout(pad=tight_layout_pad, w_pad=0.0)
# only put full xlabel after tight layout
ax1.set_xlabel('time (in units of 1/death rate)')
ax1.xaxis.set_label_coords(0.95, -0.1)

## broken axis marker
# how big to make the diagonal lines in axes coordinates
d = .02
offset = .01
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color=almostblack, clip_on=False)
ax1.plot((1-d-offset, 1+d-offset), (-d, +d), **kwargs)
ax1.plot((1-d+offset, 1+d+offset), (-d, +d), **kwargs)

#### Plot insets ####
def make_cbar(ax, vkwargs, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad="3%")
    cbarkwargs = dict(ticks=[vkwargs['vmin'], vkwargs['vmax']],
                      orientation='horizontal')
    cbar = fig.colorbar(im, cax=cax, **cbarkwargs)
    cbar.ax.tick_params(labelsize='x-small', length=0.0, pad=axis_labelpad)
    cbar.outline.set_linewidth(linewidth * 0.25)
    cbar.ax.xaxis.labelpad = - 0.6 * fontsize
    cbar_label = cbar.ax.set_xlabel(label, fontsize='x-small')
    # workaround for pdf/svg export for more smoothness
    # see matplotlib colorbar documentation
    cbar.solids.set_edgecolor("face")
    # reduce linewidth of edges to prevent overspill
    cbar.solids.set_linewidth(linewidth * 0.25)
    return cbar, cbar_label
pos = ax1.get_position(fig).get_points()
pos[1][0] = ax2.get_position(fig).get_points()[1][0]
illugrid = gridspec.GridSpec(4, 3,
                             left=pos[0, 0] + 0.08*(pos[1, 0] - pos[0, 0]),
                             bottom=pos[0, 1] + 0.27*(pos[1, 1] - pos[0, 1]),
                             right=pos[1, 0] - 0.0*(pos[1, 0] - pos[0, 0]),
                             top=pos[1, 1] - 0.0*(pos[1, 1] - pos[0, 1]),
                             wspace=1.0, hspace=2.0)
## plot P(t) insets
gridpos = [(slice(0, 2), 0), (slice(1, 3), 1), (slice(2, 4), 2)]
for i, num in enumerate(numplot):
    ax = plt.Subplot(fig, illugrid[gridpos[i]])
    fig.add_subplot(ax)
    for spine in ax.spines:
        ax.spines[spine].set_linewidth(linewidth * 0.5)
    p = ns_mf[num] / np.sum(ns_mf[num])
    vkwargs = dict(vmin=0.0, vmax=10.0)
    im = ax.imshow(p.reshape(N, N)*N**2, cmap=cm.gray_r,
                   interpolation='nearest', **vkwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar, cbar_label = make_cbar(ax, vkwargs, '$P_r(t)$')
    axannotate = ax2 if i == 2 else ax1
    axannotate.annotate('', xy=(ts_mf[num]*lambda_, costs_norm[num]),
            xytext=fig.transFigure.inverted().transform(cbar.ax.transAxes.transform(cbar_label.get_position())),
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="-", shrinkB=0, shrinkA=0.9*fontsize,
                            connectionstyle="arc3,rad=+0.0",
                            edgecolor=almostblack, linewidth=linewidth),
            )
## plot Q inset
ax = plt.Subplot(fig, illugrid[0:2, 2])
fig.add_subplot(ax)
for spine in ax.spines:
    ax.spines[spine].set_linewidth(linewidth * 0.5)
vmin, vmax = 0.0, 10.0
im = ax.imshow(Q.reshape(N, N)*N**2, cmap=cm.gray_r,
               interpolation='nearest', vmin=vmin, vmax=vmax)
cbar, cbar_label = make_cbar(ax, vkwargs, '$Q_a$')
ax.set_xticks([])
ax.set_yticks([])

#### finalize figure ####
fig.savefig('fig5.svg')
plt.show()
