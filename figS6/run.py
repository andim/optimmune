import os
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune
import lib.projgrad as projgrad

reltol = 1e-8
sigma = 1e-2
Delta = 0.1 * sigma
# strength of noise of antigen distribution
kappa = 0.5
# characteristic correlation length
xi = 10.0 * sigma
seed = 5878

x = np.arange(0.0, 1.0, Delta)
prng = np.random.RandomState(seed)
Q = immune.correlated_random_cutoff(kappa, len(x), 1.0 / xi, alpha=2.0,
                                    prng=prng)
def func(x):
    return np.exp(- np.abs(x)**2 / (2.0 * sigma**2))
convolve = immune.make_convolve_1d_pbc(func, x)
def objective(p):
    Ptilde = convolve(p)
    # to explicitely avoid divide by zero issues
    # cost infinite if Ptilde = 0 somewere, gradient non-sensical
    if np.amin(Ptilde) <= 0.0:
        return np.inf, None
    f = np.sum(Q / Ptilde)
    grad = - convolve(Q / Ptilde**2)
    return f, grad

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    popt = projgrad.minimize(objective, p0=np.ones(len(x))/len(x),
                             reltol=reltol, maxiters=1e7, algo='fast',
                             show_progress=True, nboundupdate=1e3)
    np.savez('data/res.npz', Q=Q, P=popt, Ptilde=convolve(popt), x=x)
