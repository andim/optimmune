####
# Figure S4
# needs:
# - varykernel*.npz produced by runvarykernel.py
# - fattailed*.npz produced by runfattailed.py
# - rdf_*.npz produced by calc-rdf.py
# - sq_*.npz produced by calc-rdf.py
####

import glob

import sys
sys.path.append('..')
from lib.mppaper import *
import lib.mpsetup as mpsetup
import lib.immune as immune

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(figsize_double[0]*0.8, figsize_double[1]))
gridargs = dict(left=0.05, right=0.98, top=0.95, bottom=0.08,
                wspace=0.4, hspace=0.65)
grid = gridspec.GridSpec(4, 4, **gridargs)

columns = [('varykernel_gammaa1', lambda x: np.exp(-np.abs(x))),
           ('varykernel_gammaa2', lambda x: np.exp(-np.abs(x)**2)),
           ('varykernel_gammaa4', lambda x: np.exp(-np.abs(x)**4)),
           ('fattailed', lambda x: 1.0 / (1.0 + np.abs(x)**2))]

import runvarykernel
eta = runvarykernel.eta

files = glob.glob('data/varykernel*.npz')
files.sort(key=lambda f: immune.parse_value(f, 'gammaa'))
gamma_as = sorted(list(set(immune.parse_value(f, 'gammaa') for f in files)))

## create nested list of axes
axes = []
for i in range(len(columns)):
    axes.append([])
    for j in range(4):
        ax = plt.Subplot(fig, grid[j, i])
        axes[i].append(ax)

for i, (name, kernel) in enumerate(columns):
    ## plot kernel
    ax = axes[i][0]
    x = np.linspace(-2, 2, 1000)
    ax.plot(x, kernel(x))
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 1.0])
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_ylabel('f(x)')
    ax.set_xlabel('$x \; / \; \eta$')

    ## plot popt
    ax = axes[i][1]
    npz = np.load(glob.glob('data/%s*.npz' % name)[0])
    popt = npz['P']
    N = len(npz['x'])
    vmin = 0
    vmax = 100
    im = ax.imshow(popt.reshape(N, N)*N**2, extent=(0, 1.0/eta, 0, 1.0/eta),
                   # same as nearest, but better for svg export
                   interpolation = 'none',
                   vmin = vmin, vmax = vmax)
    im.set_cmap('gray_r')
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.set_ylabel('$x \; / \;  \eta$')
    ax.set_xlabel('$y \; / \; \eta$')

    ## plot radial distribution function
    ax = axes[i][2]
    npz = np.load('data/rdf_%s.npz' % name)
    binmid, rdfsa = npz['binmid'], npz['rdfs']
    mean = np.mean(rdfsa, axis=1)
    ax.plot(binmid/eta, mean)
    ax.axhline(1.0, color=almostblack)
    ax.set_xlabel('$R \; / \; \eta$')
    ax.set_ylabel('g(R)')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 1.75)
    ax.set_yticks([0.0, 1.0])

    ## plot structure factor
    ax = axes[i][3]
    npz = np.load('data/sq_%s.npz' % name)
    bins, avpowers = npz['bins'], npz['avpowers']
    ax.plot(2*np.pi*bins[1:]*eta, avpowers[1:], '-')
    ax.axhline(1.0, color=almostblack)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 4)
    ax.set_xlabel("$q \; \eta$")
    ax.set_ylabel("$S(q)$")
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)

#### finalize figure ####
for ax in [item for sublist in axes for item in sublist]:
    ax.xaxis.labelpad = axis_labelpad
    ax.yaxis.labelpad = axis_labelpad
    fig.add_axes(ax)
for i, axl in enumerate(axes):
    for j, ax in enumerate(axl):
        if not j == 1:
            mpsetup.despine(ax)

# label different plot types
    mpsetup.label_axes([axes[0][i] for i in range(4)], xy=(0.01, 0.9),
                       xycoords=('figure fraction', 'axes fraction'),
                       fontsize = 'medium')
# label different gamma_as
for i, axl in enumerate(axes[:-1]):
    axl[0].annotate(r'\textbf{exponential}, $\boldsymbol{\gamma = %g}$' % gamma_as[i],
                    xy=(0.5, 0.99),
                    xycoords=('axes fraction', 'figure fraction'),
                    ha = 'center', va = 'top',
                    fontsize = 'medium')
axes[-1][0].annotate(r'\textbf{longtailed}', xy=(0.5, 0.99),
                     xycoords=('axes fraction', 'figure fraction'),
                     ha='center', va='top',
                     fontsize='medium')
fig.savefig('figS4.svg')
plt.show()
