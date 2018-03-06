# -*- coding: utf-8 -*-
"""

This file contains routines for simples loglikelihoods and priors for test
cases.

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# Tell module what it's allowed to import
__all__ = ["bimodal_normal_sample","bimodal_normal_lnlike",
           "bimodal_normal_lnprior","bimodal_normal_lnprob","rosenbrock_lnlike",
           "rosenbrock_lnprior","rosenbrock_sample","rosenbrock_lnprob"]

import numpy as np


################################################################################
#
# Functions for bimodal posterior
#
################################################################################


def bimodal_normal_sample(n):
    """
    Sample N points from the prior pi(x) is a uniform distribution over
    [-10, 10] x [-10, 10]

    Parameters
    ----------
    n : int
        Number of samples

    Returns
    -------
    sample : floats
        n x 2 array of floats samples from the prior
    """

    return np.random.uniform(low=-10, high=10, size=(n,2)).squeeze()
# end function



def bimodal_normal_lnlike(theta, mus=None, icovs=None):
    """
    Bimodal Gaussian log-likelihood function.

    Parameters
    ----------
    theta : array
    mus : arrays (optional)
        mean vectors of the gaussians.  Defaults to None.
    icovs : arrays (optional)
        inverse of the covariance matrices

    Returns
    -------
    l : float
        log prior
    """

    # Did the user specify the inverse of the covariance matrices?
    if icovs is None:
        cov1 = np.array([[1.0, 1.0],[-2.0, 1.0]])
        cov2 = np.array([[1.0, 0.0],[0.0, 1.0]])

        icov1 = np.linalg.inv(cov1)
        icov2 = np.linalg.inv(cov2)
    else:
        icov1 = icovs[0]
        icov2 = icovs[1]

    # Did the user specify the means?
    if mus is None:
        mu1 = np.array([-6.0,-6.0])
        mu2 = np.array([6.0,6.0])
    else:
        mu1 = mus[0]
        mu2 = mus[1]

    diff1 = np.array(theta) - mu1
    diff2 = np.array(theta) - mu2

    if diff1.ndim < 2:
        # Max function forces bimodality
        return np.max([-np.dot(diff1,np.dot(icov2,diff1.T))/2.0,
                       -np.dot(diff2,np.dot(icov2,diff2.T))/2.0])
    else:
        assert (diff1.shape == diff2.shape)

        return np.array([np.max([-np.dot(diff1[ii],np.dot(icov2,diff1[ii].T))/2.0,
                       -np.dot(diff2[ii],np.dot(icov2,diff2[ii].T))/2.0])
                       for ii in range(len(diff1))])


# end function


def bimodal_normal_lnprior(theta):
    """
    Flat log prior over [-10,10]^2 for the bimodal normal case

    Parameters
    ----------
    theta : array

    Returns
    -------
    l : float
        log prior
    """

    if np.any(np.fabs(theta) > 10):
        return -np.inf
    else:
        return 0.0
# end function


def bimodal_normal_lnprob(theta, mus=None, icovs=None):
    """
    Compute the log probability (log posterior) as likelihood * prior

    Parameters
    ----------
    theta : array
    mus : arrays (optional)
        mean vectors of the gaussians.  Defaults to None.
    icovs : arrays (optional)
        inverse of the covariance matrices

    Returns
    -------
    l : float
        log probability
    """

    # Compute prior
    lp = bimodal_normal_lnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + bimodal_normal_lnlike(theta, mus=mus, icovs=icovs)
#end function


################################################################################
#
# Functions for Rosenbrock function posterior
#
################################################################################


def rosenbrock_lnlike(x):
    """
    2D Rosenbrock function as a log likelihood following Wang & Li (2017)

    Parameters
    ----------
    x : array

    Returns
    -------
    l : float
        likelihood
    """

    x = np.array(x)
    if x.ndim > 1:
        x1 = x[:,0]
        x2 = x[:,1]
    else:
        x1 = x[0]
        x2 = x[1]

    return -0.01*(x1 - 1.0)**2 - (x1*x1 - x2)**2
# end function


def rosenbrock_lnprior(x):
    """
    Uniform log prior for the 2D Rosenbrock likelihood following Wang & Li (2017)
    where the prior pi(x) is a uniform distribution over [-5, 5] x [-5, 5]

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


def rosenbrock_sample(n):
    """
    Sample N points from the prior pi(x) is a uniform distribution over
    [-5, 5] x [-5, 5]

    Parameters
    ----------
    n : int
        Number of samples

    Returns
    -------
    sample : floats
        n x 2 array of floats samples from the prior
    """

    return np.random.uniform(low=-5, high=5, size=(n,2)).squeeze()
# end function


def rosenbrock_lnprob(theta):
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
    lp = rosenbrock_lnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + rosenbrock_lnlike(theta)
#end function
