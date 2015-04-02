####
# Figure 4
# needs:
# - data/*.npz produced by run.py
####

import glob

import sys
sys.path.append('..')
from lib.mppaper import *
import lib.mpsetup as mpsetup
import lib.immune as immune

files = sorted((immune.parse_value(f, 'epsilon'), f) for f in glob.glob("data/*.npz"))
import run
sigma = run.sigma

#### left figure ####
epsilons = []
similarities = []
similaritiesQ = []
similaritiesPtilde = []
for epsilon, f in files:
    npz = np.load(f)
    P1 = npz['P1']
    P2 = npz['P2']
    Ptilde1 = npz['Ptilde1']
    Ptilde2 = npz['Ptilde2']
    Q1 = npz['Q1']
    Q2 = npz['Q2']
    epsilons.append(epsilon)
    similarities.append(immune.similarity(P1, P2))
    similaritiesQ.append(immune.similarity(Q1, Q2))
    similaritiesPtilde.append(immune.similarity(Ptilde1, Ptilde2))

fig = plt.figure()
ax = fig.add_subplot(121)
ax.axhline(1.0, color=almostblack)
ax.plot(epsilons, similarities, label='$P_r^\star$', **linedotstyle)
ax.plot(epsilons, similaritiesQ, label='$Q_a$', **linedotstyle)
ax.plot(epsilons, similaritiesPtilde, label=r'$\tilde P_a$', **linedotstyle)
ax.set_xlabel('noise $\epsilon$')
ax.set_ylabel('Similarity')
ax.set_xlim(0.0, 0.5)
ax.set_ylim(0.0, 1.05)
ax.legend(ncol=1, loc='center right')
ax.xaxis.labelpad = axis_labelpad
ax.yaxis.labelpad = axis_labelpad
mpsetup.despine(ax)
fig.tight_layout(pad=tight_layout_pad)
fig.subplots_adjust(top=0.85)

#### right figures ####
epsilon_illustration = 0.2
epsilon, f = [tup for tup in files if tup[0] == epsilon_illustration][0]
npz = np.load(f)
P1 = npz['P1']
P2 = npz['P2']
Qbase = npz['Qbase']
Q1 = npz['Q1']
Q2 = npz['Q2']
x = npz['x']

axQ = fig.add_subplot(222)
for i, Q in enumerate([Q1, Q2]):
    axQ.plot(x/sigma, Q, lw=0.5 * linewidth, label='ind. %g' % (i+1))
axQ.set_xlim(0, 10)
axQ.set_ylabel(r'$Q_a$')

axP = fig.add_subplot(224, sharex=axQ)
for i, p in enumerate([P1, P2]):
    axP.plot(x/sigma, p, label='ind. %g' % (i+1), **linedotstyle)
axP.locator_params(axis='x', nbins=5, tight=True)
axP.set_xlim(0, 20)
axP.set_ylabel(r'$P_r^\star$')
axP.legend(ncol=2, handletextpad=0.1,
           loc='upper right',
           bbox_to_anchor=(1.05, 1.20))

for a in [axQ, axP]:
    a.set_ylim(ymin=0.0)
    mpsetup.despine(a)
    a.set_yticks([])
    a.xaxis.labelpad = axis_labelpad
    a.yaxis.labelpad = axis_labelpad
axP.set_xlabel('$x \; / \; \sigma$')
plt.setp(axQ.get_xticklabels(), visible=False)

#### finish figure ####
fig.tight_layout(pad=tight_layout_pad, h_pad=1.0)
fig.savefig('fig4.svg')
plt.show()
