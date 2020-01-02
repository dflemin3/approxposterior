# -*- coding: utf-8 -*-
"""
:py:mod:`likelihood.py` - Example Likelihood Functions
------------------------------------------------------

This file contains routines for simple test, likelihood, prior, and sampling
functions for cases like the Wang & Li (2017) Rosenbrock function example.
"""

# Tell module what it's allowed to import
__all__ = ["rosenbrockLnlike", "rosenbrockLnprior","rosenbrockSample",
           "rosenbrockLnprob", "testBOFn", "testBOFnSample", "testBOFnLnPrior",
           "sphereLnlike", "sphereSample", "sphereLnprior"]

import numpy as np
from scipy.optimize import rosen


################################################################################
#
# Functions for Rosenbrock function posterior
#
################################################################################


def rosenbrockLnlike(theta):
    """
    Rosenbrock function as a loglikelihood following Wang & Li (2017)

    Parameters
    ----------
    theta : array

    Returns
    -------
    l : float
        likelihood
    """

    return -rosen(theta)/100.0
# end function


def rosenbrockLnprior(theta):
    """
    Uniform log prior for the 2D Rosenbrock likelihood following Wang & Li (2017)
    where the prior pi(x) is a uniform distribution over [-5, 5] x [-5, 5] x ...
    for however many dimensions (dim = x.shape[-1])

    Parameters
    ----------
    theta : array

    Returns
    -------
    l : float
        log prior
    """

    if np.any(np.fabs(theta) > 5):
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


################################################################################
#
# 1D Test Function for Bayesian Optimization
#
################################################################################


def testBOFn(theta):
    """
    Simple 1D test Bayesian optimization function adapted from
    https://krasserm.github.io/2018/03/21/bayesian-optimization/
    """

    theta = np.asarray(theta)
    return -np.sin(3*theta) - theta**2 + 0.7*theta
# end function


def testBOFnSample(n=1):
    """
    Sample N points from the prior pi(x) is a uniform distribution over
    [-2, 1]

    Parameters
    ----------
    n : int (optional)
        Number of samples. Defaults to 1.

    Returns
    -------
    sample : floats
        n x 1 array of floats samples from the prior
    """

    return np.random.uniform(low=-1, high=2, size=(n,1)).squeeze()
# end function


def testBOFnLnPrior(theta):
    """
    Log prior distribution for the test Bayesian Optimization function. This
    prior is a simple uniform function over [-2, 1]

    Parameters
    ----------
    theta : float/array

    Returns
    -------
    l : float
        log prior
    """

    if np.any(theta < -1) or np.any(theta > 2):
        return -np.inf
    else:
        return 0.0
# end function


################################################################################
#
# 2D Test Function for Bayesian Optimization
#
################################################################################


def sphereLnlike(theta):
    """
    Sphere test 2D optimization function. Note: This is actually the
    negative of the sphere function and it's just a 0 mean, unit std Gaussian.
    Taken from: https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters
    ----------
    theta : array

    Returns
    -------
    val : float
        Function value at theta
    """

    theta = np.asarray(theta)
    return -np.sum(theta**2)
# end function


def sphereSample(n=1):
    """
    Sample N points from the prior pi(theta) is a uniform distribution over
    [-2, 2]

    Parameters
    ----------
    n : int (optional)
        Number of samples. Defaults to 1.

    Returns
    -------
    sample : floats
        n x 1 array of floats samples from the prior
    """

    return np.random.uniform(low=-2, high=2, size=(n,2)).squeeze()
# end function


def sphereLnprior(theta):
    """
    Log prior distribution for the sphere test optimization function.
    This prior is a simple uniform function over [-2, 2] for each dimension.

    Parameters
    ----------
    theta : float/array

    Returns
    -------
    l : float
        log prior
    """

    if np.any(np.fabs(theta) > 2):
        return -np.inf
    else:
        return 0.0
# end function
