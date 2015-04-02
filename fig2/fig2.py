####
# Figure 2
####

import sys
sys.path.append('..')
from lib.mppaper import *
import lib.mpsetup as mpsetup
import matplotlib.gridspec as gridspec

fig = plt.figure()

## plot main axes
ax_cost = fig.add_subplot(111)
# critical cross-reactivity width
sigmac = 2**.5
# value of cost function in constant area
constcost = 2

def optcost(x):
    # work-around for bug in numpy.piecewise for scalar input values
    if not hasattr(x, "__len__"):
        x = np.asarray([x])
    return np.piecewise(x, [x < sigmac, x > sigmac],
                        [lambda x: constcost, lambda x: x / (1 - 1/x**2)**.5])
xmax = sigmac * 2
ymax = 5.5
x = np.linspace(0, xmax, 100)
ax_cost.plot(x, optcost(x), c=color_cycle[0])
ax_cost.set_xticks([sigmac])
ax_cost.set_xticklabels(['$\sqrt{2}$'])
lines = ax_cost.vlines(sigmac, 0, constcost, linestyles='dotted')
ax_cost.set_ylim(0, ymax)
ax_cost.set_xlim(0, xmax)
ax_cost.set_ylabel(r'optimized cost$\; \times \; \sigma$')
ax_cost.set_xlabel('$\sigma \; / \; \sigma_Q$')
ax_cost.xaxis.set_label_coords(0.95, -0.025)
ax_cost.set_yticks([])
mpsetup.despine(ax_cost)
fig.tight_layout(pad=tight_layout_pad)

## plot inset axes
pos = ax_cost.get_position(fig).get_points()
illugrid = gridspec.GridSpec(1, 3,
                             left=pos[0, 0] + 0.03*(pos[1, 0] - pos[0, 0]),
                             bottom=pos[0, 1] + 0.61*(pos[1, 1] - pos[0, 1]),
                             right=pos[1, 0] - 0.17*(pos[1, 0] - pos[0, 0]),
                             top=pos[1, 1] - 0.03*(pos[1, 1] - pos[0, 1]))
illucolors = color_cycle[1:]
x = np.linspace(-4, 4, 100)
ymax = 0.8
for i, sigma in enumerate([0.45, 1.3, 2.1]):
    ax = plt.Subplot(fig, illugrid[0, i], axisbg='none')
    fig.add_subplot(ax)
    line_Q, = ax.plot(x, np.exp(-0.5*x**2) / (2 * np.pi)**.5,
                      color=illucolors[0], zorder=1)
    if 2 > sigma**2:
        sigma_p = (2 - sigma**2)**.5
        line_P, = ax.plot(x, np.exp(-0.5*x**2/sigma_p**2)/(sigma_p*(2*np.pi)**.5),
                          color=illucolors[1])
    else:
        ax.vlines(x=0.0, ymin=0.0, ymax=ymax, color=illucolors[1], zorder=2)
    y = 2.0 / (2 * np.pi * np.exp(1))**.5
    line_Fap, = ax.plot([-sigma, +sigma], [y, y], color=illucolors[2])
    ax.set_ylim(0, ymax)
    mpsetup.despine(ax)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax_cost.annotate('', xy=(sigma, optcost(sigma)),
                     xytext = fig.transFigure.inverted().transform(ax.transData.transform((0, 0))),
                     textcoords = 'figure fraction',
                     arrowprops = dict(arrowstyle="->",
                                       connectionstyle="arc3,rad=+0.1",
                                       facecolor=almostblack, linewidth=linewidth),
                     )
leg = ax_cost.legend((line_Q, line_P, line_Fap),
                     (r'$Q(a)$', r'$P^\star(r)$', r'$2 \sigma$'),
                     frameon=False,
                     loc='upper right',
                     bbox_to_anchor=(1.02, 1.02),
                     bbox_transform=fig.transFigure)

#### finish figure ####
fig.savefig('fig2.svg')
plt.show()
