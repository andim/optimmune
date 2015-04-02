import os
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune
import lib.projgrad as projgrad

sigma = 0.05
Delta = 0.1 * sigma
kappa = 0.25
reltol = 1e-8
# in units of 1/sigma^d
self_density = 0.04
# in units of sigma
self_exclusion_width = 1.0
seed = 1494

prng = np.random.RandomState(seed)
N = int(1.0/Delta)
# insure N is always even
N = N if N % 2 == 0 else N+1
x = np.linspace(0.0, 1.0, N, endpoint=False)
def func(x):
    return np.exp(- x**2 / (2 * sigma**2))
convolve = immune.make_convolve_2d_pbc(func, x)

# self avoidance
N_self = int(round(self_density/sigma**2))
self_antigens = prng.rand(N_self, 2)
xv, yv = np.meshgrid(x, x)
mask = np.zeros(xv.shape, dtype=np.bool)
for xp, yp in self_antigens:
    for xshift in [-1, 0, 1]:
        for yshift in [-1, 0, 1]:
            mask = np.logical_or(mask, ((xv - xp + xshift)**2 +
                                        (yv - yp + yshift)**2) <
                                       (self_exclusion_width * sigma)**2)
mask = mask.flatten()

Q = np.ones(N**2) if kappa == 0.0 else prng.lognormal(mean=0.0,
                                                      sigma=immune.sigma_lognormal_from_cv(kappa),
                                                      size=N**2)
Q /= np.sum(Q)
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
    p0 = np.ones(Q.shape)
    p0[mask] = 0.0
    p0 /= np.sum(p0)
    popt = projgrad.minimize(objective, p0, reltol=reltol, maxiters=1e7,
                             algo='fast', show_progress=True, nboundupdate=1e2,
                             mask=mask)
    np.savez('data/res.npz', Q=Q, P=popt, x=x, Ptilde=convolve(popt),
             self=self_antigens)
