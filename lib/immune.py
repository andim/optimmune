import numpy as np
import itertools
import math
import re

# scipy not available everywhere, but not needed for everything
try:
    import scipy.integrate
except:
    pass

# pyfftw can be used to speed up convolutions, but not available everywhere
try:
    import sys
    path = ['/home/andreas/repos/clones/pyFFTW/',
            '/users/mayer/repos/clones/pyFFTW/trunk/']
    path.extend(sys.path)
    sys.path = path
    import pyfftw
    pyfftw.interfaces.cache.enable()
    if not pyfftw.version == '0.10.0':
        raise Exception('Outdated pyfftw version')
    pyfftw_nthreads = 1
    use_pyfftw = True
    print 'using pyfftw'
except:
    use_pyfftw = False


def params_combination(params, nruns):
    """Make a list of all combinations of the parameters."""
    # convert float entries to 1-element list for itertools.product
    params += (range(1, nruns+1),)
    params = [[p] if isinstance(p, float) or isinstance(p, int) else p
              for p in params]
    return list(itertools.product(*params))


def parse_value(string, indicator='_'):
    """ Parses a value starting after indicator from a string. """
    return float(re.search('(?<=%s)[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
                           % indicator, string).group(0))


def correlated_random_cutoff(cv, N, q0, alpha=2.0, prng=np.random):
    # generate normal random variables
    ux = prng.normal(0.0, sigma_lognormal_from_cv(cv), N)
    uq = np.fft.rfft(ux)
    q = np.abs(np.fft.fftfreq(N, d=1./N)[:len(uq)])
    q = 2 * np.pi * q
    S = 1.0/(1.0+(q/q0)**alpha)
    S /= np.mean(S)
    etaq = S**.5 * uq
    etax = np.fft.irfft(etaq)
    lognorm = np.exp(etax)
    lognorm /= np.sum(lognorm)
    return lognorm


def correlated_random_cutoff_2d(cv, N, q0, alpha=2.0, prng=np.random):
    # generate normal random variables
    ux = prng.normal(0.0, sigma_lognormal_from_cv(cv), (N, N))
    uq = np.fft.fft2(ux)
    q = np.abs(np.fft.fftfreq(N, d=1./N)[:uq.shape[0]])
    q = 2 * np.pi * q
    q1, q2 = np.meshgrid(q, q)
    S = 1.0/(1.0+((q1**2 + q2**2)**.5/q0)**alpha)
    S /= np.mean(S)
    etaq = S**.5 * uq
    etax = np.real(np.fft.ifft2(etaq))
    lognorm = np.exp(etax)
    lognorm /= np.sum(lognorm)
    return lognorm


def correlated_random(alpha, N, std):
    """ generate correlated random time series by fourier filtering technique.

        alpha: S(q) = q^(-alpha)
        N : length of time series
        std : std**2 = power for low q
    """
    # sampling frequency
    sfreq = N
    # generate normal random variables
    ux = np.random.normal(0.0, 1.0, N)
    uq = np.fft.rfft(ux)
    q = np.fft.fftfreq(N, d=1./sfreq)[:len(uq)]
    q[-1] = np.abs(q[-1])
    q = 2 * np.pi * q / N
    S = np.ones_like(q)
    S[1:] = q[1:]**(-alpha)
    S[0] = S[1]
    S /= S[0]
    S *= std * N
    etaq = S**.5 * uq
    etax = np.fft.irfft(etaq)
    lognorm = np.exp(etax)
    lognorm /= np.sum(lognorm)
    return lognorm


def integrate_popdyn(Q, convolve, A, g, lambda_, n0, ts, t0=0.0,
                     integrator='lsoda',
                     integrator_kwargs={},
                     callback=None):
    """
        Simulate mean-field population dynamics.

        Q : pathogen distribution,
        convolve : function for convolution with cross-reactivity Kernel
        A : availability function
        g : \overline{F}
        n0 : initial population
        ts : time points at which to store results
        t0 : initial time
        integrator : scipy.ode integrator
        integrator_kwargs : scipy.ode integrator keyword arguments
        lambda_ : free parameter of population dynamics
        callback : function to be called at each time point
    """

    def f(t, n):
        ntilde = convolve(n)
        return n * (convolve(Q * A(ntilde)) - lambda_)

    def cost(n):
        p = n / np.sum(n)
        return np.sum(Q * g(convolve(p)))

    ns = np.empty((len(ts), len(n0)))
    costs = np.empty((len(ts),))

    r = scipy.integrate.ode(f)
    r.set_integrator(integrator, **integrator_kwargs)
    r.set_initial_value(n0, t0)
    for num in range(len(ts)):
        r.integrate(ts[num])
        if not r.successful():
            print 'Integration not succesful'
        if np.amin(r.y) < 0.0:
            print 'Integration violated non-negativity'
        ns[num] = r.y
        costs[num] = cost(r.y)
        if callback:
            callback(ts[num], ns[num], costs[num])
    return ns, costs


def integrate_popdyn_stoch(Q, frp, A, g, lambda_, n0, tend, dt=1e0, nsave=1,
                           callback=None, prng=np.random):
    """
        Simulate stochastic population dynamics.

        Q : pathogen distribution,
        convolve : function for convolution with cross-reactivity Kernel
        A : availability function
        g : \overline{F}
        lambda_ : free parameter of population dynamics
        dt : time interval that each pathogen is present
        nsave : how often to save
        callback : function to be called at each time point
    """

    def cost(n):
        p = n / np.sum(n)
        return np.sum(Q * g(np.dot(p, frp)))

    n = np.copy(n0)
    nsteps = int(math.ceil(tend / dt))

    nsavetot = nsteps / nsave
    ns = np.empty((nsavetot, len(n0)))
    costs = np.empty((nsavetot,))

    inds = prng.choice(len(Q), size=nsteps, p=Q)

    def f(i, n):
        ind = inds[i]
        return n * (A(np.sum(frp[ind] * n)) * frp[ind] - lambda_)

    for i in range(nsteps):
        if i % nsave == 0:
            ns[i / nsave] = n
            costs[i / nsave] = cost(n)
        n += dt * f(i, n)
        if callback:
            callback(i*dt, n, cost(n))
    ts = np.arange(0, tend, dt * nsave)
    return ts, ns, costs


def make_convolve_1d_pbc(func, x, B=1.0):
    frp = np.zeros_like(x)
    for shift in np.arange(-5.0, 5.1, B):
        frp += func(x+shift)
    frp_ft = np.fft.rfft(frp)

    def convolve(x):
        return np.fft.irfft(np.fft.rfft(x) * frp_ft)
    return convolve


def make_convolve_2d_pbc(func, x, B=1.0):
    N = len(x)
    # indexing = ij only available in new numpy versions
    # xv, yv = np.meshgrid(x, x, indexing='ij')
    # this is equivalent
    yv, xv = np.meshgrid(x, x)
    frp = np.zeros_like(xv)
    for xshift in np.arange(-5.0, 5.1, B):
        for yshift in np.arange(-5.0, 5.1, B):
            frp += func(((xv+xshift)**2 + (yv+yshift)**2)**.5)

    if use_pyfftw:
        fft = pyfftw.builders.rfft2(frp)
        frp_ft = fft(frp)
        ifft = pyfftw.builders.irfft2(frp_ft, threads=pyfftw_nthreads)
        fft = pyfftw.builders.rfft2(frp, threads=pyfftw_nthreads)

        def convolve(x):
            return ifft(fft(x.reshape((N, N))) * frp_ft).flatten()
    else:
        frp_ft = np.fft.rfft2(frp)
        fft = np.fft.rfft2
        ifft = np.fft.irfft2

        def convolve(x):
            return ifft(fft(x.reshape((N, N))) * frp_ft).flatten()
    return convolve


def build_1d_frp_matrix(func, x, sigma, B=1):
    """ Builds quadratic frp matrix respecting pbc.

    func: Kernel function
    x: position of points
    sigma: width of Kernel
    """
    N = len(x)
    A = np.zeros((N, N))
    shifts = np.arange(-5, 6) * B
    for r in range(N):
        for p in range(N):
            value = 0
            for shift in shifts:
                value += func(x[r] - x[p] + shift, sigma[r])
            A[r, p] = value
    return A


# ported from numpy github as not yet available in installed numpy version
def rfftfreq(n, d=1.0):
    if not (isinstance(n, int) or isinstance(n, np.integer)):
        raise ValueError("n should be an integer")
    val = 1.0/(n*d)
    N = n//2 + 1
    results = np.arange(0, N, dtype=int)
    return results * val


class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        # if args is not Hashable we can't cache
        # easier to ask for forgiveness than permission
        try:
            if args in self.cache:
                return self.cache[args]
            else:
                value = self.func(*args)
                self.cache[args] = value
                return value
        except TypeError:
            return self.func(*args)


def distance_pbc(p1, p2):
    """ Calculate distance respecting periodic boundary conditions. """
    total = 0
    try:
        coord_pairs = zip(p1, p2)
    except:
        coord_pairs = ((p1, p2),)
    for i, (a, b) in enumerate(coord_pairs):
        delta = abs(b - a)
        if delta > 1.0 - delta:
            delta = 1.0 - delta
        total += delta ** 2
    return total ** 0.5


def azimuthalAverage(image, center=None, stddev=False, returnradii=False,
                     return_nr=False, binsize=0.5, weights=None, steps=False,
                     interpnan=False, left=None, right=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).
    stddev - if specified, return the azimuthal std dev instead of average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - return weighted average instead of simple average if keyword
              parameteris set. weights.shape must = image.shape.
              weighted stddev is undefined, so don't set weights and stddev.
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...

    Modified from https://github.com/keflavich/agpy
    (MIT licensed by Adam Ginsburg)
    """

    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)
    nbins = int(np.round(r.max() / binsize))+1
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins+1)
    # but we're probably more interested in the bin centers
    # than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # Find out which radial bin each point in the map belongs to
    whichbin = np.digitize(r.flat, bins)

    # recall that bins are from 1 to nbins
    # (expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        radial_prof = np.array([image.flat[whichbin == b].std()
                                for b in xrange(1, nbins+1)])
    else:
        radial_prof = np.array([(image*weights).flat[whichbin == b].sum() /
                                weights.flat[whichbin == b].sum()
                                for b in xrange(1, nbins+1)])

    if interpnan:
        radial_prof = np.interp(bin_centers,
                                bin_centers[radial_prof == radial_prof],
                                radial_prof[radial_prof == radial_prof],
                                left=left, right=right)

    if returnradii:
        return bin_centers, radial_prof
    elif return_nr:
        # how many per bin (i.e., histogram)?
        # there are never any in bin 0, because the lowest index
        # returned by digitize is 1
        nr = np.bincount(whichbin)[1:]
        return nr, bin_centers, radial_prof
    else:
        return radial_prof


def calc_rdf_1d(p, x, nbins=50, epsilon=0.0):
    """ Calculate radial distribution function in 1d. """
    nonzero = np.nonzero(p) if epsilon == 0.0 else p > epsilon
    p_nz = p[nonzero]
    pos = x[nonzero]

    pdists = []
    weights = []
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            pdists.append(distance_pbc(pos[i], pos[j]))
            weights.append(p_nz[i] * p_nz[j])
    hist, bins = np.histogram(pdists, bins=np.linspace(0, 0.5, nbins),
                              weights=weights)
    binmid, rdf = bins[:-1]+0.5*np.diff(bins), hist / np.diff(bins)
    return binmid, rdf


def calc_rdf_2d(p, xv, yv, nbins=50, distlim=0.5, epsilon=0.0):
    """ Calculate radial distribution function in 2d. """
    nonzero = np.nonzero(p) if epsilon == 0.0 else p > epsilon
    pos = np.dstack((xv[nonzero], yv[nonzero]))[0]
    p_nz = p[nonzero]

    pdists = []
    weights = []
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            pdist = distance_pbc(pos[i], pos[j])
            if pdist < distlim:
                pdists.append(pdist)
                weights.append(p_nz[i] * p_nz[j])
    hist, bins = np.histogram(pdists, bins=nbins, weights=weights)
    binmid = bins[:-1]+0.5*np.diff(bins)
    rdf = 2 * hist / (np.pi * (bins[1:]**2 - bins[:-1]**2))
    return binmid, rdf


def similarity(x, y):
    "sum_r P1_r P2_r / sqrt[(sum_r P1_r^2) x (sum_r P2_r^2)]"
    return np.sum(x*y) / (np.sum(x**2) * np.sum(y**2))**.5


def sigma_lognormal_from_cv(cv):
    """ Lognormal parameter sigma from coefficient of variation. """
    return (np.log(cv**2 + 1))**.5
