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
    x = npz['x']
    xv, yv = np.meshgrid(x, x, indexing='ij')
    N = len(x)
    P = npz['P'].reshape((N, N))
    binmid, rdf = immune.calc_rdf_2d(P, xv, yv)
    np.savez('data/rdf_%s' % os.path.basename(f)[4:], binmid=binmid, rdf=rdf)
