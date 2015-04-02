import glob
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune

globarg_name = [('fattailed*', 'fattailed'),
                ('varykernel*gammaa1*', 'varykernel_gammaa1'),
                ('varykernel*gammaa2*', 'varykernel_gammaa2'),
                ('varykernel*gammaa4*', 'varykernel_gammaa4')]

for globarg, name in globarg_name:
    files = glob.glob('data/' + globarg)
    pfts = []
    for f in files:
        npz = np.load(f)
        x = npz['x']
        N = len(x)
        popt = npz['P'].reshape((N, N))
        pft = np.fft.fft2(popt) / np.sum(popt**2)**.5
        pfts.append(np.fft.fftshift(pft))
    powers = np.abs(np.asarray(pfts))**2
    bins, avpowers = immune.azimuthalAverage(np.mean(powers, axis=0),
                                             binsize=1.0, returnradii=True)
    np.savez('data/sq_%s' % name, bins=bins, avpowers=avpowers)
