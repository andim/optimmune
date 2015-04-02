import os
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune
import lib.projgrad as projgrad

# range of cross-reactivity
eta = 0.05
# discretization
Delta = 0.1 * eta
# cv of antigen distribution
kappa = 0.25
reltol = 1e-8
nruns = 10
# random number generator seed
seed = np.random.randint(1e4)

paramscomb = immune.params_combination((eta, Delta, kappa, reltol), nruns)

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    if len(sys.argv) > 1:
        njob = int(sys.argv[1])
        eta, Delta, kappa, reltol, nrun = paramscomb[njob-1]
    else:
        print "# parameter sets", len(paramscomb)
        sys.exit()

    N = int(1.0/Delta)
    # insure N is always even
    N = N if N % 2 == 0 else N+1
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    def func(x):
        return 1.0 / (1.0 + (np.abs(x)/eta)**2)
    convolve = immune.make_convolve_2d_pbc(func, x)
    prng = np.random.RandomState(seed)
    Q = np.ones(N**2) if kappa == 0.0 else prng.lognormal(mean=0.0,
                                                          sigma=immune.sigma_lognormal_from_cv(kappa),
                                                          size=N**2)
    Q /= np.sum(Q)
    def objective(p):
        Ptilde = convolve(p)
        f = np.sum(Q / Ptilde)
        grad = - convolve(Q / Ptilde**2)
        return f, grad

    popt = projgrad.minimize(objective, np.ones(N**2)/N**2, reltol=reltol,
                                 maxiters=1e7, show_progress=False,
                                 nboundupdate=1e2)
    np.savez('data/fattailed_seed%g.npz' % seed,
             Q=Q, P=popt, x=x, Ptilde=convolve(popt))
