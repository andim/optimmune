####
# Figure S5
# needs: data/*.npz produces by run.npz
####

import glob

import sys
sys.path.append('..')
from lib.mppaper import *
import lib.mpsetup as mpsetup
import lib.immune as immune

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(figsize_double[0]*0.8, figsize_double[1]*0.8))
gridargs = dict(left=0.09, right=0.98, top=0.95, bottom=0.1, wspace=0.45, hspace=0.4)
grid = gridspec.GridSpec(3, 3, **gridargs)


files = glob.glob('data/opt*.npz')
files.sort(key=lambda f: immune.parse_value(f, 'scatter'))
scatters = sorted(list(set(immune.parse_value(f, 'scatter') for f in files)))
import run
sigma = run.sigma

axes = []
for i in range(len(scatters)):
    axes.append([])
    for j in range(3):
        ax = plt.Subplot(fig, grid[j, i])
        axes[i].append(ax)

for i, scatter in enumerate(scatters):
    # plot popt
    ax = axes[i][0]
    npz = np.load(glob.glob('data/opt*scatter%g*.npz' % scatter)[0])
    P = npz['P']
    x = npz['x']
    ax.plot(x/sigma, P)
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 0.11)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.set_ylabel('$P(r)$')
    ax.set_xlabel('$r \; / \; \sigma$')

    # plot radial distribution function
    ax = axes[i][1]
    npz = np.load('data/rdf_scatter%g.npz' % scatter)
    binmid, rdfsa = npz['binmid'], npz['rdfs']
    mean = np.mean(rdfsa, axis=1)
    ax.plot(binmid/sigma, mean)
    ax.axhline(1.0, color=almostblack)
    ax.set_xlabel('$R \; / \; \sigma$')
    ax.set_ylabel('g(R)')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 2.8)
    ax.locator_params(axis='y', nbins=5)

    # plot structure factor
    ax = axes[i][2]
    npz = np.load('data/sq_scatter%g.npz' % scatter)
    bins, avpowers = npz['bins'], npz['avpowers']
    ax.plot(2*np.pi*bins[1:]*sigma, avpowers[1:], '-')
    ax.axhline(1.0, color=almostblack)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 5.0)
    ax.set_xlabel("$q \; \sigma$")
    ax.set_ylabel("$S(q)$")
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)

#### finalize figure ####
for ax in [item for sublist in axes for item in sublist]:
    ax.xaxis.labelpad = axis_labelpad
    ax.yaxis.labelpad = axis_labelpad
    fig.add_axes(ax)
# despine axes
for i, axl in enumerate(axes):
    for j, ax in enumerate(axl):
        mpsetup.despine(ax)
# label different plot types
mpsetup.label_axes([axes[0][i] for i in range(3)], xy=(0.01, 0.9),
                   xycoords=('figure fraction', 'axes fraction'),
                   fontsize='medium')
# label different scatters
for i, axl in enumerate(axes):
    axl[0].annotate(r'$\boldsymbol{\mathrm{scatter} = %g}$' % scatters[i],
                    xy=(0.5, 0.99), xycoords=('axes fraction', 'figure fraction'),
                    ha='center', va='top', fontsize='medium')
# save fig
fig.savefig('figS5.svg')
plt.show()
