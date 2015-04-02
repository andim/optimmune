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
    rdfs = []
    for f in files:
        npz = np.load(f)
        x = npz['x']
        xv, yv = np.meshgrid(x, x, indexing='ij')
        N = len(x)
        P = npz['P'].reshape((N, N))
        binmid, rdf = immune.calc_rdf_2d(P, xv, yv)
        rdfs.append(rdf)
    rdfsa = np.asarray(rdfs).T
    np.savez('data/rdf_%s' % name, binmid=binmid, rdfs=rdfsa)
