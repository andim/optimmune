import os
import numpy as np

import sys
sys.path.append('..')
import lib.immune as immune
import lib.projgrad as projgrad
import lib.cimmune as cimmune

#### Parameters ####
sigma = 0.1
Delta = 0.5 * sigma
beta = 1.0 / 110.
seed = 3863

# scatter in Q
kappa = 1e0
# correlation length
xi = 5.0 * sigma

# scatter in initial condition
kappa_initial = 2e0
# correlation length
xi_initial = 5.0 * sigma

N = int(1.0/Delta)
# insure N is always even
N = N if N % 2 == 0 else N+1
x = np.linspace(0.0, 1.0, N, endpoint=False)

# death rate
lambda_ = 0.001
# total initial population size
ntot0 = 110.0

# from where to where to integrate population dynamics
tstart = 1e0
tend = 1e6
# time step of stochastic dynamics
dt = 5e0
nsave = 1

# relative tolerance of optimization
reltol = 1e-8

## functional form of competition and cost
g = lambda x: beta / (beta + x)
gdiff = lambda x: - beta / (beta + x)**2
A = lambda x: 1.0/(1.0 + x)**2

#### run simulation ####
if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    ## setup
    factor = 1.0 / (2.0 * sigma**2)

    @immune.memoized
    def func_a(x):
        return np.exp(- factor * x**2)
    prng = np.random.RandomState(seed)
    Q = immune.correlated_random_cutoff_2d(kappa, N, 1.0 / xi, alpha=2.0,
                                           prng=prng).flatten()

    ## find optimal repertoire to provide baseline cost
    convolve = immune.make_convolve_2d_pbc(func_a, x)

    def objective(p):
        Ptilde = convolve(p)
        f = np.sum(Q * g(Ptilde))
        grad = convolve(Q * gdiff(Ptilde))
        return f, grad
    popt = projgrad.minimize(objective, np.ones(N**2)/N**2, reltol=reltol,
                             maxiters=1e7, show_progress=True,
                             nboundupdate=1e3)
    fopt = np.sum(Q * g(convolve(popt)))

    ## solve mean field equations
    n0 = immune.correlated_random_cutoff_2d(kappa_initial, N, 1.0 / xi_initial,
                                            alpha=2.0, prng=prng).flatten()
    n0 *= ntot0
    ts_mf = np.logspace(np.log10(tstart), np.log10(tend), 100)
    ns_mf, costs_mf = immune.integrate_popdyn(Q, convolve, A, g, lambda_,
                                              n0, ts_mf)

    ## solve three independent realizations of the stochastic equations
    frp = cimmune.build_2d_frp_matrix(func_a, x)
    ts_stoch, ns_stoch, costs_stoch = immune.integrate_popdyn_stoch(Q, frp, A, g, lambda_,
                                            n0, tend, dt=dt, prng=prng, nsave=nsave)
    ts_stoch, ns_stoch2, costs_stoch2 = immune.integrate_popdyn_stoch(Q, frp, A, g, lambda_,
                                            n0, tend, dt=dt, prng=prng, nsave=nsave)
    ts_stoch, ns_stoch3, costs_stoch3 = immune.integrate_popdyn_stoch(Q, frp, A, g, lambda_,
                                            n0, tend, dt=dt, prng=prng, nsave=nsave)

    ## save data
    np.savez('data/out.npz', Q=Q, x=x,
             popt=popt, fopt=fopt,
             ns_mf=ns_mf, ts_mf=ts_mf, costs_mf=costs_mf,
             ts_stoch=ts_stoch,
             # ns_stoch=ns_stoch, ns_stoch2=ns_stoch2, ns_stoch3=ns_stoch3,
             costs_stoch=costs_stoch, costs_stoch2=costs_stoch2, costs_stoch3=costs_stoch3
             )
