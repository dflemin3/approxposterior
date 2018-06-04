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

def setup_gp(theta, y, gp):
    """
    Initialize a george GP object.  Utility function for creating a new GP when
    the data its conditioned on changes sizes, i.e. when a new point is added

    Parameters
    ----------
    theta : array
    y : array
        data to condition GP on
    gp : george.GP
        Gaussian Process that learns the likelihood conditioned on forward
        model input-output pairs (theta, y)

    Returns
    -------
    new_gp : george.GP
    """

    # Create GP using same kernel, updated estimate of the mean, but new theta
    new_gp = george.GP(kernel=gp.kernel, fit_mean=True,
                       mean=np.nanmedian(y))
    new_gp.compute(theta)

    return new_gp
# end function
