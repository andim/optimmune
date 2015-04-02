# needs results from run.py

import glob
import os.path
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune

files = glob.glob('data/opt*.npz')
for f in files:
    npz = np.load(f)
    N = len(npz['x'])
    P = npz['P'].reshape((N, N))
    pft = np.fft.fft2(P)
    pft = np.fft.fftshift(pft)
    power = np.abs(np.asarray(pft))**2
    power_norm = power / np.sum(P**2)
    bins, sq = immune.azimuthalAverage(power_norm, binsize=1.0, returnradii=True)
    bins, psd = immune.azimuthalAverage(power, binsize=1.0, returnradii=True)
    np.savez('data/sq_%s' % os.path.basename(f)[4:], bins=bins, sq=sq, psd=psd)
