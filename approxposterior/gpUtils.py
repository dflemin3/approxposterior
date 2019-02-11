# -*- coding: utf-8 -*-
"""

Gaussian process utility functions.

"""

# Tell module what it's allowed to import
__all__ = ["optimizeGP"]

from . import pool
from . import utility as util
import numpy as np
import george
from scipy.optimize import minimize, basinhopping


def _nll(p, gp, y):
    """
    Given parameters and data, compute the negative log likelihood of the data
    under the george Gaussian process.

    Parameters
    ----------
    p : array
        GP hyperparameters
    gp : george.GP
    y : array
        data to condition GP on

    Returns
    -------
    nll : float
        negative log-likelihood of y under gp
    """

    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25
# end function


def _grad_nll(p, gp, y):
    """
    Given parameters and data, compute the gradient of the negative log
    likelihood of the data under the george Gaussian process.

    Parameters
    ----------
    p : array
        GP hyperparameters
    gp : george.GP
    y : array
        data to condition GP on

    Returns
    -------
    gnll : float
        gradient of the negative log-likelihood of y under gp
    """

    # Negative gradient of log likelihood
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)
# end function


def defaultGP(theta, y):
    """
    Basic utility function that initializes a simple GP that works well in many
    applications.

    Parameters
    ----------
    theta : array
        Design points
    y : array
        Data to condition GP on, e.g. the lnlike * lnprior at each design point,
        theta.

    Returns
    -------
    gp : george.GP
        Gaussian process with initialized kernel and factorized covariance matrix.
    """

    # Guess initial metric, or scale length of the covariances in loglikelihood space
    initialMetric = np.array([5.0*len(theta)**(-1.0/theta.shape[-1]) for _ in range(theta.shape[-1])])

    # Create kernel: We'll model coveriances in loglikelihood space using a
    # Squared Expoential Kernel as we anticipate Gaussian-ish posterior
    # distributions in our 2-dimensional parameter space
    metric_bounds = ((-10, 10) for _ in range(theta.shape[-1]))
    kernel = george.kernels.ExpSquaredKernel(initialMetric,
                                             ndim=theta.shape[-1],
                                             metric_bounds=metric_bounds)

    # Guess initial mean function
    mean = np.mean(y)

    # Create GP and compute the kernel, factor the covariance matrix
    gp = george.GP(kernel=kernel, fit_mean=True, mean=mean)
    gp.compute(theta)

    return gp
# end function


def optimizeGP(gp, theta, y, seed=None, nRestarts=5, method=None, options=None,
               p0=None, nCores=1):
    """

    Optimize hyperparameters of an arbitrary george Gaussian Process kenerl
    using either a straight-up maximizing the log-likelihood or k-fold cv in which
    the log-likelihood is maximized for each fold and the best one is chosen.

    Note that the cross-validation used here is sort of cross valiation.
    Instead of training on training set and evaluating the model on the test
    set, we do both on the training set.  That is a cardinal sin of ML, but we
    do that because matrix shape sizes and evaluating the log-likelihood of the
    data requires it.

    Parameters
    ----------
    gp : george.GP
    theta : array
    y : array
        data to condition GP on
    seed : int (optional)
        numpy RNG seed.  Defaults to None.
    nRestarts : int (optional)
        Number of times to restart the optimization.  Defaults to 5. Increase
        this number if the GP isn't optimized well.
    method : str (optional)
        scipy.optimize.minimize method.  Defaults to nelder-mead if None.
    options : dict (optional)
        kwargs for the scipy.optimize.minimize function.  Defaults to None, or
        an empty dictionary.
    p0 : array (optional)
        Initial guess for kernel hyperparameters.  If None, defaults to
        ndim values randomly sampled from a uniform distribution over [-10, 10)
    nCores : int (optional)
        If > 1, use multiprocessing to distribute optimization restarts

    Returns
    -------
    optimized_gp : george.GP
    """

    # Set default parameters if None are provided
    if method is None:
        method = "nelder-mead"
    if options is None:
        options = {"adaptive" : True}

    # Run the optimization routine n_restarts times, maybe using multiprocessing
    res = []
    mll = []

    # Figure out how many cores to use with InterruptiblePool
    if nCores > 1:
        poolType = "MultiPool"
    else:
        poolType = "SerialPool"

    # Use multiprocessing to distribution optimization calls
    with pool.Pool(pool=poolType, processes=nCores) as optPool:

        # Inputs for each process
        if p0 is None:
            iterables = [(_nll, np.hstack(([np.mean(y)], [np.random.uniform(low=-10, high=10) for _ in range(theta.shape[-1])]))) for _ in range(nRestarts)]
        else:
            iterables = [(_nll, np.array(p0) + 1.0e-3 * np.random.randn(len(p0))) for _ in range(nRestarts)]


        # keyword arguments for minimizer
        mKwargs = {"jac" : _grad_nll,
                   "args" : (gp, y),
                   "method" : method,
                   "options" : options,
                   "bounds" : None}

        # Run the minimization on nCores
        fn = util.functionWrapperArgsOnly(minimize, **mKwargs)
        results = optPool.map(fn, iterables)

    # Extract solutions and recompute marginal log likelihood for solutions
    for result in results:
        res.append(result.x)

        # Update the kernel
        gp.set_parameter_vector(result.x)
        gp.recompute()

        # Compute marginal log likelihood
        mll.append(gp.log_likelihood(y, quiet=True))

    # Pick result with largest marginal log likelihood
    ind = np.argmax(mll)

    # Update gp
    gp.set_parameter_vector(res[ind])
    gp.recompute()

    return gp
# end function
