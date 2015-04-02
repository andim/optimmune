####
# Figure S6
# needs:
# - data/res.npz produced by run.py
####

import sys
sys.path.append('..')
from lib.mppaper import *

#### load data ####
npz = np.load('data/res.npz')
Q = npz['Q']
x = npz['x']
P = npz['P']
Ptilde = npz['Ptilde']

## load parameter
import run
sigma = run.sigma
Delta = run.Delta

#### plot ####
fig = plt.figure()

# Plot Pstar
ax = fig.add_subplot(111)
ax.plot(x/sigma, P, label=r'$P^\star_r$', c=color_cycle[0])
ax.set_ylim(0.0, 0.07)
ax.set_xlabel('$x \; / \;  \sigma$')
ax.set_ylabel(r'$P^\star_r$')
ax.locator_params(axis='y', nbins=4, tight=True)
ax.xaxis.labelpad = axis_labelpad
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')

# Plot Q and tildePstar on twin axes
ax2 = ax.twinx()
ax2.plot(x/sigma, Ptilde / ((2*np.pi)**.5 / (Delta/sigma)),
         label=r'$\tilde P^\star_r / \int f$', c=color_cycle[1])
ax2.plot(x/sigma, Q, label=r'$Q_a$', c=color_cycle[2])
ax2.set_ylim(ymin=0.0)
ax2.set_ylabel(r'$Q_a$ and $\tilde P^\star_r / \int f$')
ax2.locator_params(axis='y', nbins=4, tight=True)

# make legend for both axes
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, ncol=3,
          loc='upper center', bbox_to_anchor=(0.5, 1),
          bbox_transform=fig.transFigure)

#### Finalize figure ####
fig.tight_layout(pad=tight_layout_pad)
fig.savefig('figS6.svg')
plt.show()
