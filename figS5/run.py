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
kappa = 0.25
reltol = 1e-8
nruns = 30
scatter = [0.01, 0.1, 1.0]
seed = np.random.randint(1e4)

paramscomb = immune.params_combination((sigma, Delta, kappa, reltol, scatter), nruns)


#### run optimization ####
if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    if len(sys.argv) > 1:
        njob = int(sys.argv[1])
        sigma, Delta, kappa, reltol, scatter, nrun = paramscomb[njob-1]
    else:
        print "# parameter sets", len(paramscomb)
        sys.exit()
    prng = np.random.RandomState(seed)
    N = int(1.0/Delta)
    # insure N is always even
    N = N if N % 2 == 0 else N+1
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    @immune.memoized
    def func(x, sigma):
        return np.exp(- x**2 / (2.0 * sigma**2)) / sigma
    sig = immune.sigma_lognormal_from_cv(scatter)
    sigmas = prng.lognormal(mean=np.log(sigma) - 0.5 * sig**2, sigma=sig,
                            size=x.shape)
    frp = immune.build_1d_frp_matrix(func, x, sigmas)
    Q = np.ones(N) if kappa == 0.0 else prng.lognormal(mean=0.0,
                                                       sigma=immune.sigma_lognormal_from_cv(kappa),
                                                       size=N)
    Q /= np.sum(Q)
    def objective(p):
        Ptilde = np.dot(p, frp)
        f = np.sum(Q / Ptilde)
        grad = - np.dot(frp, Q / Ptilde**2)
        return f, grad
    popt = projgrad.minimize(objective, np.ones(len(x))/len(x), reltol=reltol,
                             maxiters=1e8, show_progress=False,
                             nboundupdate=1e2, algo='fast')
    np.savez('data/opt_scatter%g_seed%i.npz' % (scatter, seed),
             Q=Q, P=popt, x=x, Ptilde=np.dot(popt, frp), sigmas=sigmas)
