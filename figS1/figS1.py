####
# Figure S1
# needs:
# - data/*.npz produced by run.py
####

import glob

import sys
sys.path.append('..')
from lib.mppaper import *
import lib.mpsetup as mpsetup
import lib.immune as immune

#### import data ####
paramname = 'Delta'
files = glob.glob('data/Delta*.npz')
files_decorated = sorted([(immune.parse_value(f, paramname), f) for f in files])
# extract sorted list of varying parameter
params = sorted(list(set(immune.parse_value(f, paramname) for f in files)))
import run
sigma = run.sigma

#### Plot figure ####
fig = plt.figure()
ax = fig.add_subplot(111)
for i, Delta in enumerate(params):
    this_file = [f for f in files_decorated if f[0] == Delta][0][1]
    npz = np.load(this_file)
    Q = npz['Q']
    popt = npz['P']
    x = npz['x']
    color = color_cycle[i]
    ax.plot(x/sigma, popt/Delta, '-', c=color, label='%g' % (Delta/sigma))
    ax.set_xlabel('$r \;  / \; \sigma$')
    ax.set_xlim(0, 25)
    ax.set_ylabel('$P^\star_r \; / \; \Delta$')
    ax.legend(title='$\Delta \; / \; \sigma$', ncol=1, loc='upper right')

#### finalize figure ####
fig.tight_layout(pad=tight_layout_pad)
mpsetup.despine(ax)
fig.savefig('figS1.svg')
plt.show()
