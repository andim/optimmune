import os
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune
import lib.projgrad as projgrad


# express all lengths scales in terms of the size B of the box
# range of cross-reactivity
sigma = 0.005
# cv of antigen distribution
kappa = 0.5
# characteristic correlation length
xi = 10.0 * sigma
# strength of noise per individual
epsilon = np.arange(0.0, 0.5, 0.025)

# discretization
delta = 0.05 * sigma
# optimization algorithm parameter
optlibparams = dict(abstol=0.0, reltol=1e-8, maxiters=1e6, show_progress=False,
                    nboundupdate=100, algo='fast')
# random number generator seed
seed = np.random.randint(1e4)

#### run ####
if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    # number of job in queue
    if len(sys.argv) > 1:
        njob = int(sys.argv[1])
        epsilon = epsilon[njob-1]
    else:
        print "# parameter sets:", len(epsilon)
        sys.exit()

    prng = np.random.RandomState(seed)
    N = int(1.0/delta)
    # insure N is always even
    N = N if N % 2 == 0 else N+1
    Qbase = immune.correlated_random_cutoff(kappa, N, 1.0 / xi, alpha=2.0)

    def func_a(x):
        return np.exp(- np.abs(x)**2 / (2.0 * sigma**2))

    x = np.linspace(0, 1, N, endpoint=False)
    convolve = immune.make_convolve_1d_pbc(func_a, x)
    p0 = np.ones(np.prod(Qbase.shape))/np.prod(Qbase.shape)

    def objective(p, Q):
        Ptilde = convolve(p)
        # to explicitely avoid divide by zero issues
        # cost infinite, gradient non-sensical
        if np.amin(Ptilde) <= 0.0:
            return np.inf, None
        f = np.sum(Q / Ptilde)
        grad = - convolve(Q / Ptilde**2)
        return f, grad

    Q1 = Qbase * (1.0 if epsilon == 0.0 else prng.lognormal(mean=0.0,
                                                            sigma=epsilon,
                                                            size=Qbase.shape))
    Q1 /= np.sum(Q1)
    objective1 = lambda p: objective(p, Q1)
    P1 = projgrad.minimize(objective1, p0=p0, **optlibparams)
    Ptilde1 = convolve(P1)

    Q2 = Qbase * (1.0 if epsilon == 0.0 else prng.lognormal(mean=0.0,
                                                            sigma=epsilon,
                                                            size=Qbase.shape))
    Q2 /= np.sum(Q2)
    objective2 = lambda p: objective(p, Q2)
    P2 = projgrad.minimize(objective2, p0=p0, **optlibparams)
    Ptilde2 = convolve(P2)

    np.savez('data/res_epsilon%g_seed%g.npz' % (epsilon, seed),
             Qbase=Qbase, x=x,
             Q1=Q1, Q2=Q2,
             P1=P1, P2=P2,
             Ptilde1=Ptilde1, Ptilde2=Ptilde2)
