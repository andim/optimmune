####
# Figure S2
# needs:
# - rdf*.npz produced by calc-rdf.py
# - sq*.npz produced by calc-sq.py
####

import glob

import sys
sys.path.append('..')
from lib.mppaper import *
import lib.mpsetup as mpsetup
import lib.immune as immune

import matplotlib.gridspec as gridspec

thisfigsize = figsize
thisfigsize[1] *= 0.75
fig = plt.figure(figsize=thisfigsize)
grid = gridspec.GridSpec(1, 2, left=0.06, right=0.97, top=0.91, bottom=0.2,
        wspace=0.4, hspace=0.35)
labeled_axes = []

#### Subfigure A: g(R) ####
## import data
filenames = glob.glob('data/rdf*kappa1*')
rdfsum = None
for filename in filenames:
    npz = np.load(filename)
    if rdfsum is not None:
        rdfsum += npz['rdf']
    else:
        rdfsum = npz['rdf']
rdf = rdfsum / len(filenames)
binmid = npz['binmid']
sigma = immune.parse_value(filename, 'sigma')
## plot
ax = fig.add_subplot(121)
labeled_axes.append(ax)
ax.plot(binmid/sigma, rdf)
ax.axhline(1.0, color=almostblack)
ax.set_xlim(0, 8)
ax.locator_params(axis='y', nbins=3, tight=True)
ax.set_ylim(0, 1.8)
ax.set_xlabel('$R$ / $\sigma$')
ax.xaxis.labelpad = axis_labelpad
ax.yaxis.labelpad = axis_labelpad
ax.set_ylabel('radial distribution\nfunction $g(R)$')
mpsetup.despine(ax)

#### Subfigure B: S(q) ####
ax = fig.add_subplot(122)
labeled_axes.append(ax)
kappas = sorted(list(set(immune.parse_value(f, 'kappa') for f in glob.glob('data/sq*'))))[::-1]
for kappa in kappas:
    files = glob.glob('data/sq*kappa%g*' % kappa)
    sqsum = None
    for f in files:
        npz = np.load(f)
        if sqsum is not None:
            sqsum += npz['sq']
        else:
            sqsum = npz['sq']
    sq = sqsum / len(files)
    sigma = immune.parse_value(f, 'sigma')
    kappa = immune.parse_value(f, 'kappa')
    Delta = immune.parse_value(f, 'Delta')
    N = (1.0 / (Delta * sigma))**2
    bins = npz['bins']
    ax.plot(2*np.pi*bins[1:]*sigma, sq[1:], label='$\kappa$ = %g' % kappa)
ax.axhline(1.0, color=almostblack)
ax.set_xlim(0, 30)
ax.set_ylim(0, 4.7)
ax.set_ylabel("norm. power spectral\ndensity $S(q)$")
legend = ax.legend(loc='upper right', fontsize='small')
ax.set_xlabel("$q \; \sigma$")
ax.xaxis.labelpad = axis_labelpad
ax.yaxis.labelpad = axis_labelpad
ax.locator_params(axis='y', nbins=5)
mpsetup.despine(ax)

#### Finalize figure ####
fig.tight_layout(pad=tight_layout_pad)
labelstyle = r'{\sf \textbf{%s}}'
mpsetup.label_axes(labeled_axes, labels='ABCD', labelstyle=labelstyle,
                   xy=(-0.12, 0.92), fontsize='medium',
                   xycoords=('axes fraction'), fontweight='bold')
fig.savefig('figS2.svg')
plt.show()
