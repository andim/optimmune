import os
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune
import lib.projgrad as projgrad

# range of cross-reactivity
eta = 0.05
# crossreactivity Kernel exponent
gamma_a = [1.0, 2.0, 4.0]
# cv of antigen distribution
kappa = 0.25
# discretization
Delta = 0.1 * eta
nruns = 10
optlibparams = dict(abstol=0.0, reltol=1e-8, maxiters=1e9, show_progress=False,
                    nboundupdate=100, algo='fast')
# random number generator seed
seed = np.random.randint(1e4)

paramscomb = immune.params_combination((eta, gamma_a, Delta, kappa), nruns)

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    if len(sys.argv) > 1:
        njob = int(sys.argv[1])
        eta, gamma_a, Delta, kappa, nrun = paramscomb[njob-1]
    else:
        print "# parameter sets", len(paramscomb)
        sys.exit()

    N = int(1.0/Delta)
    # insure N is always even
    N = N if N % 2 == 0 else N+1
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    def func(x):
        return np.exp(- (np.abs(x) / eta)**gamma_a)
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

    popt = projgrad.minimize(objective, np.ones(N**2)/N**2, **optlibparams)
    np.savez('data/varykernel_gammaa%g_seed%i.npz' % (gamma_a, seed),
             Q=Q, P=popt, x=x, Ptilde=convolve(popt))
