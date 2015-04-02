import glob
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune

paramname = 'scatter'
files = glob.glob('data/opt*.npz')
files_decorated = sorted([(immune.parse_value(f, paramname), f) for f in files])
# extract sorted list of varying parameter
params = sorted(list(set(immune.parse_value(f, paramname) for f in files)))

for param in params[::-1]:
    this_files = [f for f in files_decorated if f[0] == param]
    rdfs = []
    for param, f in this_files:
        npz = np.load(f)
        x = npz['x']
        P = npz['P']
        binmid, rdf = immune.calc_rdf_1d(P, x, epsilon=1e-6, nbins=50)
        rdfs.append(rdf)
    rdfsa = np.asarray(rdfs).T
    np.savez('data/rdf_%s%g' % (paramname, param), binmid=binmid, rdfs=rdfsa)
