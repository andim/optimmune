import glob, os.path
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
    powers = []
    for param, f in this_files:
        npz = np.load(f)
        x = npz['x']
        N = len(x)
        P = npz['P']
        pft = np.fft.rfft(P)
        power =  np.abs(np.asarray(pft))**2
        power_norm = power / np.sum(P**2)
        powers.append(power_norm)
    avpowers = np.mean(powers, axis=0)
    np.savez('data/sq_%s%g' % (paramname, param),
             bins=np.fft.rfftfreq(len(P), d=np.mean(np.diff(x))), avpowers=avpowers)
