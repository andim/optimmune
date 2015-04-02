import os
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune
import lib.projgrad as projgrad

sigma = 0.05
Delta = 0.1 * sigma
kappa = 1.0
reltol = 1e-8
seed = 2610

N = int(1.0/Delta)
# insure N is always even
N = N if N % 2 == 0 else N+1
x = np.linspace(0.0, 1.0, N, endpoint=False)
def func(x):
    return np.exp(- x**2 / (2.0 * sigma**2))
convolve = immune.make_convolve_2d_pbc(func, x)
prng = np.random.RandomState(seed)
if kappa == 0.0:
    Q = np.ones(N**2)
else:
    Q = prng.lognormal(mean=0.0,
                       sigma=immune.sigma_lognormal_from_cv(kappa),
                       size=N**2)
Q /= np.sum(Q)
def objective(p):
    Ptilde = convolve(p)
    f = np.sum(Q / Ptilde)
    grad = - convolve(Q / Ptilde**2)
    return f, grad

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    popt = projgrad.minimize(objective, np.ones(N**2)/N**2, reltol=reltol,
                             maxiters=1e7, show_progress=True,
                             nboundupdate=1e2)
    np.savez('data/2d.npz', Q=Q, P=popt, x=x, Ptilde=convolve(popt))
