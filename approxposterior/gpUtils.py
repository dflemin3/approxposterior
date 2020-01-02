# -*- coding: utf-8 -*-
"""
:py:mod:`gpUtils.py` - Gaussian Process Utilities
-------------------------------------------------

Gaussian process utility functions for initializing GPs and optimizing their
hyperparameters.

"""

# Tell module what it's allowed to import
__all__ = ["defaultHyperPrior", "defaultGP", "optimizeGP"]

from . import utility as util
import numpy as np
import george
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def defaultHyperPrior(p):
    """
    Default prior function for GP hyperparameters. This prior also keeps the
    hyperparameters within a reasonable huge range, [-20, 20]. Note that george
    operates on the *log* hyperparameters, except for the mean function.

    Parameters
    ----------
    p : array/iterable
        Array of GP hyperparameters

    Returns
    -------
    prior : float
    """

    # Restrict range of hyperparameters (ignoring mean term)
    if np.any(np.fabs(p)[1:] > 20):
        return -np.inf

    return 0.0
# end function


def _nll(p, gp, y, priorFn=None):
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
    priorFn : callable
        Prior function for the GP hyperparameters, p

    Returns
    -------
    nll : float
        negative log-likelihood of y under gp
    """

    # Apply priors on GP hyperparameters
    if priorFn is not None:
        if not np.isfinite(priorFn(p)):
            return np.inf

    # Catch singular matrices
    try:
        gp.set_parameter_vector(p)
    except np.linalg.LinAlgError:
        return np.inf

    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else np.inf
# end function


def _grad_nll(p, gp, y, priorFn=None):
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
    priorFn : callable
        Prior function for the GP hyperparameters, p

    Returns
    -------
    gnll : float
        gradient of the negative log-likelihood of y under gp
    """

    # Apply priors on GP hyperparameters
    if priorFn is not None:
        if not np.isfinite(priorFn(p)):
            return np.inf

    # Negative gradient of log likelihood
    return -gp.grad_log_likelihood(y, quiet=True)
# end function


def defaultGP(theta, y, order=None, white_noise=-12, fitAmp=False):
    """
    Basic utility function that initializes a simple GP with an ExpSquaredKernel.
    This kernel  works well in many applications as it effectively enforces a
    prior on the smoothness of the function and is infinitely differentiable.

    Parameters
    ----------
    theta : array
        Design points
    y : array
        Data to condition GP on, e.g. the lnlike + lnprior at each design point,
        theta.
    order : int (optional)
        Order of PolynomialKernel to add to ExpSquaredKernel. Defaults to None,
        that is, no PolynomialKernel is added and the GP only uses the
        ExpSquaredKernel
    white_noise : float (optional)
        From george docs: "A description of the logarithm of the white noise
        variance added to the diagonal of the covariance matrix". Defaults to
        ln(white_noise) = -12. Note: if order is not None, you might need to
        set the white_noise to a larger value for the computation to be
        numerically stable, but this, as always, depends on the application.
    fitAmp : bool (optional)
        Whether or not to include an amplitude term. Defaults to False.

    Returns
    -------
    gp : george.GP
        Gaussian process with initialized kernel and factorized covariance
        matrix.
    """

    # Tidy up the shapes and determine dimensionality
    theta = np.asarray(theta).squeeze()
    y = np.asarray(y).squeeze()
    if theta.ndim <= 1:
        ndim = 1
    else:
        ndim = theta.shape[-1]

    # Guess initial metric, or scale length of the covariances (must be > 0)
    initialMetric = np.fabs(np.random.randn(ndim))

    # Create kernel: We'll model coveriances in loglikelihood space using a
    # ndim-dimensional Squared Expoential Kernel
    kernel = george.kernels.ExpSquaredKernel(metric=initialMetric,
                                             ndim=ndim)

    # Include an amplitude term?
    if fitAmp:
        kernel = np.var(y) * kernel

    # Add a linear regression kernel of order order?
    # Use a meh guess for the amplitude and for the scale length (log(gamma^2))
    if order is not None:
        kernel = kernel + (np.var(y)/10.0) * george.kernels.LinearKernel(log_gamma2=initialMetric[0],
                                                                         order=order,
                                                                         bounds=None,
                                                                         ndim=ndim)

    # Create GP and compute the kernel, aka factor the covariance matrix
    gp = george.GP(kernel=kernel, fit_mean=True, mean=np.median(y),
                   white_noise=white_noise, fit_white_noise=False)
    gp.compute(theta)

    return gp
# end function


def optimizeGP(gp, theta, y, seed=None, nGPRestarts=1, method="powell",
               options=None, p0=None, gpHyperPrior=defaultHyperPrior):
    """
    Optimize hyperparameters of an arbitrary george Gaussian Process by
    maximizing the marginal loglikelihood.

    Parameters
    ----------
    gp : george.GP
    theta : array
    y : array
        data to condition GP on
    seed : int (optional)
        numpy RNG seed.  Defaults to None.
    nGPRestarts : int (optional)
        Number of times to restart the optimization.  Defaults to 1. Increase
        this number if the GP isn't optimized well.
    method : str (optional)
        scipy.optimize.minimize method.  Defaults to powell.
    options : dict (optional)
        kwargs for the scipy.optimize.minimize function.  Defaults to None.
    p0 : array (optional)
        Initial guess for kernel hyperparameters.  If None, defaults to
        np.random.randn for each parameter
    gpHyperPrior : str/callable (optional)
        Prior function for GP hyperparameters. Defaults to the defaultHyperPrior fn.
        This function asserts that the mean must be negative and that each log
        hyperparameter is within the range [-20,20].

    Returns
    -------
    optimizedGP : george.GP
    """

    # Run the optimization routine nGPRestarts times
    res = []
    mll = []

    # Optimize GP hyperparameters by maximizing marginal log_likelihood
    for ii in range(nGPRestarts):
        # Initialize inputs for each minimization
        if p0 is None:
            # Pick random guesses for kernel hyperparameters
            x0 = [np.median(y)] + [np.random.randn() for _ in range(len(gp.get_parameter_vector())-1)]
        else:
            # Take user-supplied guess and slightly perturb it
            x0 = np.array(p0) + np.min(p0) * 1.0e-3 * np.random.randn(len(p0))

        # Minimize GP nll, save result, evaluate marginal likelihood
        if method not in ["nelder-mead", "powell", "cg"]:
            jac = _grad_nll
        else:
            jac = None

        resii = minimize(_nll, x0, args=(gp, y, gpHyperPrior), method=method,
                         jac=jac, bounds=None, options=options)["x"]
        res.append(resii)

        # Update the kernel with solution for computing marginal loglike
        gp.set_parameter_vector(resii)
        gp.recompute()

        # Compute marginal log likelihood for this set of kernel hyperparameters
        mll.append(gp.log_likelihood(y, quiet=True))

    # Pick result with largest marginal log likelihood
    ind = np.argmax(mll)

    # Update gp
    gp.set_parameter_vector(res[ind])
    gp.recompute()

    return gp
# end function
