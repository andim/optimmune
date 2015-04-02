import os
import numpy as np
import scipy.signal

import sys
sys.path.append('..')
import lib.immune as immune
import lib.projgrad as projgrad

#### Parameters ####
reltol = 1e-8
sigma = 1e-2
Deltas = [1e-3, 2e-3, 4e-3]
seed = 2236

def func(x):
    return np.exp(- np.abs(x)**2 / (2.0 * sigma**2))

#### run optimization ####
if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    prng = np.random.RandomState(seed)
    ## define Q at high resolution
    Delta = 1e-3
    x = np.arange(0, 1.0, Delta)
    Qorig = prng.lognormal(mean=0.0,
                           sigma=immune.sigma_lognormal_from_cv(0.25),
                           size=x.shape)
    np.savez('data/orig.npz', Q=Qorig, x=x)

    ## solve downsampled problems
    for Delta in Deltas:
        x = np.arange(0, 1.0, Delta)
        Q = scipy.signal.resample(Qorig, len(x))
        Q /= np.sum(Q)
        convolve = immune.make_convolve_1d_pbc(func, x)

        def objective(p):
            Ptilde = convolve(p)
            f = np.sum(Q / Ptilde)
            grad = - convolve(Q / Ptilde**2)
            return f, grad
        popt = projgrad.minimize(objective, np.ones(len(x))/len(x),
                                 reltol=reltol, maxiters=1e7,
                                 show_progress=False, nboundupdate=1e3)
        np.savez('data/Delta%.5f.npz' % Delta, Q=Q, P=popt, x=x)
