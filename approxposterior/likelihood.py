# -*- coding: utf-8 -*-
"""
:py:mod:`likelihood.py` - Example Likelihood Functions
------------------------------------------------------

This file contains routines for simple loglikelihood and prior functions for
test cases, like the Wang & Li (2017) Rosenbrock function example.
"""

# Tell module what it's allowed to import
__all__ = ["rosenbrockLnlike", "rosenbrockLnprior","rosenbrockSample",
           "rosenbrockLnprob"]

import numpy as np
from scipy.optimize import rosen


################################################################################
#
# Functions for Rosenbrock function posterior
#
################################################################################


def rosenbrockLnlike(x):
    """
    Rosenbrock function as a loglikelihood following Wang & Li (2017)

    Parameters
    ----------
    x : array

    Returns
    -------
    l : float
        likelihood
    """

    return -rosen(x)/100.0
# end function


def rosenbrockLnprior(x):
    """
    Uniform log prior for the 2D Rosenbrock likelihood following Wang & Li (2017)
    where the prior pi(x) is a uniform distribution over [-5, 5] x [-5, 5] x ...
    for however many dimensions (dim = x.shape[-1])

    Parameters
    ----------
    x : array

    Returns
    -------
    l : float
        log prior
    """

    if np.any(np.fabs(x) > 5):
        return -np.inf
    else:
        return 0.0
# end function


def rosenbrockSample(n=1, dim=2):
    """
    Sample N points from the prior pi(x) is a uniform distribution over
    [-5, 5] x [-5, 5]

    Parameters
    ----------
    n : int (optional)
        Number of samples. Defaults to 1.
    dim : int (optional)
        Dimensionality. Defaults to 2.

    Returns
    -------
    sample : floats
        n x 2 array of floats samples from the prior
    """

    return np.random.uniform(low=-5, high=5, size=(n,dim)).squeeze()
# end function


def rosenbrockLnprob(theta):
    """
    Compute the log probability (log posterior) as likelihood * prior

    Parameters
    ----------
    theta : array

    Returns
    -------
    l : float
        log probability
    """

    # Compute prior
    lp = rosenbrockLnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + rosenbrockLnlike(theta)
#end function
