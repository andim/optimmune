import os
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune
import lib.projgrad as projgrad

# range of cross-reactivity
sigma = 0.05
# discretization
Delta = 0.1 * sigma
# cv of antigen distribution
kappa = [1.0, 0.25, 0.0625]
reltol = 1e-8
nruns = 30
# random number generator seed
seed = np.random.randint(1e4)

paramscomb = immune.params_combination((sigma, Delta, kappa, reltol), nruns)

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    # number of job in queue
    if len(sys.argv) > 1:
        njob = int(sys.argv[1])
        sigma, Delta, kappa, reltol, nrun = paramscomb[njob-1]
    else:
        print "# parameter sets", len(paramscomb)
        sys.exit()

    N = int(1.0/Delta)
    # insure N is always even
    N = N if N % 2 == 0 else N+1
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    def func(x):
        return np.exp(- x**2 / (2.0 * sigma**2))
    convolve = immune.make_convolve_2d_pbc(func, x)
    prng = np.random.RandomState(seed)

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
    popt = projgrad.minimize(objective, np.ones(N**2)/N**2, reltol=reltol,
                             maxiters=1e7, show_progress=False,
                             nboundupdate=1e2)
    np.savez('data/opt_sigma%g_Delta%g_kappa%g_reltol%g_seed%g.npz' % (sigma, Delta/sigma, kappa, reltol, seed),
             Q=Q, P=popt, x=x, Ptilde=convolve(popt))
