# -*- coding: utf-8 -*-
"""

Gaussian process utility functions.

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# Tell module what it's allowed to import
__all__ = ["setup_gp","optimize_gp"]

import numpy as np
import george
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, ParameterGrid
from scipy.optimize import minimize


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

    gp.set_parameter_vector(p)

    # Negative gradient of log likelihood
    ngr = -2.0 * gp.grad_log_likelihood(y, quiet=True) / \
           np.sqrt(np.exp(gp.get_parameter_vector()))

    return ngr
# end function


def optimize_gp(gp, theta, y, seed=None, n_restarts=10):
    """
    TODO: implement n_restarts

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
    n_restarts : int (optional)
        Number of times to restart the optimization.  Defaults to 10.

    Returns
    -------
    optimized_gp : george.GP
    """

    # Optimize GP by maximizing log-likelihood

    # Run the optimization routine
    p0 = gp.get_parameter_vector()
    results = minimize(_nll, p0, jac=_grad_nll, args=(gp, y), method="bfgs")

    # Update the kernel
    gp.set_parameter_vector(results.x)
    gp.recompute()

    return gp
# end function

"""
def setup_gp(theta, y, which_kernel="ExpSquaredKernel", mean=None, seed=None,
             initial_metric=None):
    Initialize a george GP object

    Parameters
    ----------
    theta : array
    y : array
        data to condition GP on
    which_kernel : str (optional)
        Name of the george kernel you want to use.  Defaults to ExpSquaredKernel.
        Options: ExpSquaredKernel, ExpKernel, Matern32Kernel, Matern52Kernel
    mean : scalar, callable (optional)
        specifies the mean function of the GP using a scalar or a callable fn.
        Defaults to None.  If None, estimates the mean as np.mean(y).
    seed : int (optional)
        numpy RNG seed.  Defaults to None.
    initial_metric : array (optional)
        Initial guess for the GP metric.  Defaults to None and is estimated to
        be the squared mean of theta.  In general, you should
        provide your own!

    Returns
    -------
    gp : george.GP

    # Guess the bandwidth
    if initial_metric is None:
        initial_metric = np.nanmedian(np.array(theta)**2, axis=0)/10

    # Which kernel?
    if str(which_kernel).lower() == "expsquaredkernel":
        kernel = george.kernels.ExpSquaredKernel(initial_metric,
                                                 ndim=np.array(theta).shape[-1])
    elif str(which_kernel).lower() == "expkernel":
        kernel = george.kernels.ExpKernel(initial_metric,
                                          ndim=np.array(theta).shape[-1])
    elif str(which_kernel).lower() == "matern32kernel":
        kernel = george.kernels.Matern32Kernel(initial_metric,
                                          ndim=np.array(theta).shape[-1])
    elif str(which_kernel).lower() == "matern52kernel":
        kernel = george.kernels.Matern52Kernel(initial_metric,
                                          ndim=np.array(theta).shape[-1])
    else:
        avail = "Available kernels: ExpSquaredKernel, ExpKernel, Matern32Kernel, Matern52Kernel"
        raise NotImplementedError("Error: Available kernels: %s" % avail)

    # Guess the mean value if nothing is given as the nanmeadian
    if mean is None:
        mean = np.nanmedian(np.array(y), axis=0)

    # Create the GP conditioned on theta
    gp = george.GP(kernel=kernel, fit_mean=True, mean=mean)
    gp.compute(theta)

    return gp
# end function
"""
