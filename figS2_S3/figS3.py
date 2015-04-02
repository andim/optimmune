####
# Figure S3
# needs:
# - sq*.npz produced by calc-sq.py
####

import glob

import sys
sys.path.append('..')
from lib.mppaper import *
import lib.mpsetup as mpsetup
import lib.immune as immune

fig = plt.figure()
axscaled = fig.add_subplot(111)
filenamebase = 'data/sq*'
kappas = sorted(list(set(immune.parse_value(f, 'kappa') for f in glob.glob(filenamebase))))[::-1]
for kappa in kappas:
    files = glob.glob(filenamebase + 'kappa%g*' % kappa)
    psdsum = None
    for f in files:
        npz = np.load(f)
        if psdsum is not None:
            psdsum += npz['psd']
        else:
            psdsum = npz['psd']
    psd = psdsum / len(files)
    sigma = immune.parse_value(f, 'sigma')
    kappa = immune.parse_value(f, 'kappa')
    Delta = immune.parse_value(f, 'Delta')
    N = (1.0 / (Delta * sigma))**2
    bins = npz['bins']
    axscaled.plot(2*np.pi*bins[1:]*sigma, psd[1:] / kappa**2, '.',
                  label='$\kappa$ = %g' % kappa)
ylims = (1e-6, axscaled.get_ylim()[1] * 2.0)
xlims = (0.1, axscaled.get_xlim()[1])
qsigma = np.logspace(-1, 1, 200)
axscaled.plot(qsigma, np.exp(qsigma**2) / (4 * N),
              label=r'$\frac{\kappa^2}{4 N}\exp\left[(q \sigma)^2\right]$')
axscaled.set_yscale('log')
axscaled.set_xscale('log')
axscaled.set_ylim(*ylims)
axscaled.set_xlim(*xlims)
axscaled.set_ylabel("power spectral density / $\kappa^2$")
h, l = axscaled.get_legend_handles_labels()
legend_kappa = axscaled.legend(h[:3], l[:3], loc='upper left',
                               title='pathogen\nheterogeneity')
legend_function = axscaled.legend(h[3:], l[3:], loc='lower right')
axscaled.add_artist(legend_kappa)
axscaled.set_xlabel("$q \; \sigma$")
axscaled.xaxis.labelpad = axis_labelpad

mpsetup.despine(axscaled)
fig.tight_layout(pad=tight_layout_pad)
fig.savefig('figS3.svg')
plt.show()
