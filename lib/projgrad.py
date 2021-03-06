import time
import numpy as np


def project(x, mask=None):
    """ Take a vector x (with possible nonnegative entries and non-normalized)
        and project it onto the unit simplex.

        mask:   do not project these entries
                project remaining entries onto lower dimensional simplex
    """
    if mask is not None:
        mask = np.asarray(mask)
        xsorted = np.sort(x[~mask])[::-1]
        # remaining entries need to sum up to 1 - sum x[mask]
        sum_ = 1.0 - np.sum(x[mask])
    else:
        xsorted = np.sort(x)[::-1]
        # entries need to sum up to 1 (unit simplex)
        sum_ = 1.0
    lambda_a = (np.cumsum(xsorted) - sum_) / np.arange(1.0, len(xsorted)+1.0)
    for i in xrange(len(lambda_a)-1):
        if lambda_a[i] >= xsorted[i+1]:
            astar = i
            break
    else:
        astar = -1
    p = np.maximum(x-lambda_a[astar],  0)
    if mask is not None:
        p[mask] = x[mask]
    return p


def minimize(objective, p0,
             nboundupdate=100,
             reltol=1e-4, abstol=0.0, maxiters=1e7,
             algo='fast',
             show_progress=False,
             callback=None,
             full_return=False,
             mask=None):
    """
    minimize     objective(P_r)
    subject to   0.0 <= P_r
                 sum(P_r) = 1

    parameters
    ----------
    objective : function returning cost, gradient
    p0 : starting guess
    nboundupdate : number of iteration between lower bound updates
    reltol, abstol, maxiters: numerical parameter
    algo: fast or normal algorithm
    show_progress: print status information during the run
    full_return: also return dict with information about optimization
    mask: Boolean array with directions along which not to optimize

    output
    ------
    optimal solution
    """

    if mask is not None:
        mask = np.asarray(mask)

        def mobjective(x):
            f, grad = objective(x)
            if grad is not None:
                grad[mask] = 0.0
            return f, grad
        mobjective = mobjective
        mproject = lambda p: project(p, mask)
    else:
        mobjective = objective
        mproject = project
    # initialize p from function input
    p = mproject(np.asarray(p0))
    # counting variable for number of iterations
    k = 0
    # lower bound for the cost function
    low = 0.0

    # setup for accelerated algorithm
    if algo == 'fast':
        y = p
        f, grad = mobjective(p)
        # starting guess for gradient scaling parameter 1 / | nabla f |
        s = 1.0 / np.linalg.norm(grad)
        # refine by backtracking search
        while True:
            y_new = mproject(y - s * grad)
            # abs important as values close to machine precision
            # might become negative in fft convolution screwing
            # up cost calculations
            f_new, grad_new = mobjective(y_new)
            if f_new < f + np.dot(y_new - y, grad.T) + \
                    0.5 * np.linalg.norm(y_new - y)**2 / s:
                break
            s *= 0.8
        # reduce s by some factor as optimal s might become smaller during
        # the course of optimization
        s /= 3.0

    told = time.time()
    while k < maxiters:
        k += 1
        f, grad = mobjective(p)

        # update lower bound on cost function
        # initialize at beginning (k=1) and then every nboundupdateth iteration
        if (k % nboundupdate == 0) or (k == 1):
            if mask is not None:
                i = np.argmin(grad[~mask])
                low = max((low, f - np.sum(p * grad) + grad[~mask][i]))
            else:
                i = np.argmin(grad)
                low = max((low, f - np.sum(p * grad) + grad[i]))
            gap = f - low
            if callback:
                callback(f, p)
            if show_progress:
                print '%g: f %e, gap %e, relgap %e' % (k, f, gap, gap/low if low != 0 else np.inf)
            if ((low != 0 and gap/low < reltol) or gap < abstol):
                if show_progress:
                    print 'stopping criterion reached'
                break

        if algo == 'fast':
            f, grad = mobjective(y)
            p, pold = mproject(y - s * grad), p
            y = p + k/(k+3.0) * (p - pold)
        else:
            # generate feasible direction by projection
            s = 0.1
            d = mproject(p - s * grad) - p
            # Backtracking line search
            deriv = np.dot(grad.T, d)
            alpha = 0.1
            # in (0, 0.5)
            p1 = 0.2
            # in (0, 1)
            p2 = 0.25
            fnew, grad = mobjective(p + alpha * d)
            while fnew > f + p1*alpha*deriv:
                alpha *= p2
                fnew, grad = mobjective(p + alpha * d)
            p += alpha * d

    else:
        print 'warning: maxiters reached before convergence'
    if show_progress:
        print 'niters %e, t per iteration %e' % (k, (time.time() - told) / k)
        print 'cost %e, low %e, gap %e, relgap %e' % (f, low, gap, gap/low if low != 0 else np.inf)

    if not full_return:
        return p
    return p, dict(niter=k, f=f, low=low)
