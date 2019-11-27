# -*- coding: utf-8 -*-
"""
:py:mod:`utility.py` - Utility Functions
----------------------------------------

Utility functions in terms of usefulness, e.g. minimizing GP utility functions
or computing KL divergences, and the GP utility functions, e.g. the bape utility.
"""

# Tell module what it's allowed to import
__all__ = ["logsubexp", "AGPUtility", "BAPEUtility", "NaiveUtility",
           "minimizeObjective", "klNumerical", "latinHypercubeSampling"]

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
from pyDOE import lhs


################################################################################
#
# Data set initialization functions
#
################################################################################


def latinHypercubeSampling(n, bounds, criterion="maximin"):
    """
    Initialize a data set of size n via latin hypercube sampling over bounds
    using pyDOE.

    Parameters
    ----------
    n : int
        Number of samples in training set
    bounds : tuple/iterable
        Parameter bounds
    criterion : str (optional)
        From the pyDOE docs:
        criterion: a string that tells lhs how to sample the points
        “center” or “c”: center the points within the sampling intervals
        “maximin” or “m”: maximize the minimum distance between points,
        but place the point in a randomized location within its interval
        “centermaximin” or “cm”: same as “maximin” but centered within the
        intervals
        “correlation” or “corr”: minimize the maximum correlation coefficient
        Defaults to "maximin"

    Returns
    -------
    samps : numpy array
        n x ndim array of initial conditions
    """

    # Extract dimensionality
    ndim = len(bounds)

    # Generate latin hypercube in each dimension over [0,1]
    samps = lhs(ndim, samples=n, criterion=criterion)

    # Scale to bounds of each dimension, return
    for ii in range(ndim):
        samps[:,ii] = (bounds[ii][1] - bounds[ii][0]) * samps[:,ii] + bounds[ii][0]

    return samps
# end function


################################################################################
#
# Define math functions
#
################################################################################


def klNumerical(x, p, q):
    """
    Estimate the KL-Divergence between pdfs p and q via Monte Carlo intergration
    using x, samples from p.

    KL ~ 1/n * sum_{i=1,n}(log (p(x_i)/q(x_i)))

    For our purposes, q is the current estimate of the pdf while p is the
    previous estimate.  This method is the only feasible method for large
    dimensions.

    See Hershey and Olsen, "Approximating the Kullback Leibler
    Divergence Between Gaussian Mixture Models" for more info

    Note that this method can result in D_kl < 0 but it's the only method with
    guaranteed convergence properties as the number of samples (len(x)) grows.
    Also, this method is shown to have the lowest error, on average
    (see Hershey and Olsen).

    Parameters
    ----------
    x : array
        Samples drawn from p
    p : function
        Callable previous estimate of the density
    q : function
        Callable current estimate of the density

    Returns
    -------
    kl : float
        KL divergence
    """
    try:
        res = np.sum(np.log(p(x)/q(x)))/len(x)
    except ValueError:
        errMsg = "ERROR: inf/NaN encountered.  q(x) = 0 likely occured."
        raise ValueError(errMsg)

    return res
# end function


def logsubexp(x1, x2):
    """
    Numerically stable way to compute log(exp(x1) - exp(x2))

    logsubexp(x1, x2) -> log(exp(x1) - exp(x2))

    Parameters
    ----------
    x1 : float
    x2 : float

    Returns
    -------
    logsubexp(x1, x2)
    """

    if x1 <= x2:
        return -np.inf
    else:
        return x1 + np.log(1.0 - np.exp(x2 - x1))
# end function


################################################################################
#
# Define utility functions
#
################################################################################


def AGPUtility(theta, y, gp, priorFn):
    """
    AGP (Adaptive Gaussian Process) utility function, the entropy of the
    posterior distribution. This is what you maximize to find the next x under
    the AGP formalism. Note here we use the negative of the utility function so
    minimizing this is the same as maximizing the actual utility function.

    See Wang & Li (2017) for derivation/explaination.

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object
    priorFn : function
        Function that computes lnPrior probability for a given theta.

    Returns
    -------
    util : float
        utility of theta under the gp
    """

    # If guess isn't allowed by prior, we don't care what the value of the
    # utility function is
    if not np.isfinite(priorFn(theta)):
        return np.inf

    # Only works if the GP object has been computed, otherwise you messed up
    if gp.computed:
        mu, var = gp.predict(y, theta.reshape(1,-1), return_var=True)
    else:
        raise RuntimeError("ERROR: Need to compute GP before using it!")

    try:
        util = -(mu + 1.0/np.log(2.0*np.pi*np.e*var))
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return util
# end function


def BAPEUtility(theta, y, gp, priorFn):
    """
    BAPE (Bayesian Active Posterior Estimation) utility function.  This is what
    you maximize to find the next theta under the BAPE formalism.  Note here we
    use the negative of the utility function so minimizing this is the same as
    maximizing the actual utility function.  Also, we log the BAPE utility
    function as the log is monotonic so the minima are equivalent.

    See Kandasamy et al. (2015) for derivation/explaination.

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object
    priorFn : function
        Function that computes lnPrior probability for a given theta.

    Returns
    -------
    util : float
        utility of theta under the gp
    """

    # If guess isn't allowed by prior, we don't care what the value of the
    # utility function is
    if not np.isfinite(priorFn(theta)):
        return np.inf

    # Only works if the GP object has been computed, otherwise you messed up
    if gp.computed:
        mu, var = gp.predict(y, theta.reshape(1,-1), return_var=True)
    else:
        raise RuntimeError("ERROR: Need to compute GP before using it!")

    try:
        util = -((2.0 * mu + var) + logsubexp(var, 0.0))
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return util
# end function


def NaiveUtility(theta, y, gp, priorFn):
    """
    Naive utility function that is maximized by GP predictions with
    large loglikelihoods and large uncertainties.

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object
    priorFn : function
        Function that computes lnPrior probability for a given theta.

    Returns
    -------
    util : float
        utility of theta under the gp
    """

    # If guess isn't allowed by prior, we don't care what the value of the
    # utility function is
    if not np.isfinite(priorFn(theta)):
        return np.inf

    # Only works if the GP object has been computed, otherwise you messed up
    if gp.computed:
        mu, var = gp.predict(y, theta.reshape(1,-1), return_var=True)
    else:
        raise RuntimeError("ERROR: Need to compute GP before using it!")

    try:
        util = -mu * var
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return util
# end function


def minimizeObjective(fn, y, gp, sampleFn, priorFn, nMinObjRestarts=5):
    """
    Find point that minimizes fn for a gaussian process gp conditioned on y,
    the data, and is allowed by the prior, priorFn.  PriorFn is required as it
    helps to select against points with non-finite likelihoods, e.g. NaNs or
    infs.  This is required as the GP can only train on finite values.

    Parameters
    ----------
    fn : function
        function to minimize that expects x, y, gp as arguments aka fn looks
        like fn_name(x, y, gp).  See *_utility functions above for examples.
    y : array
        y values to condition the gp prediction on.
    gp : george GP object
    sampleFn : function
        Function to sample initial conditions from.
    priorFn : function
        Function that computes lnPrior probability for a given theta.
    nMinObjRestarts : int (optional)
        Number of times to restart minimizing -utility function to select
        next point to improve GP performance.  Defaults to 5.  Increase this
        number of the point selection is not working well.

    Returns
    -------
    theta : (1 x n_dims)
        point that minimizes fn
    """

    # Arguments for the utility function
    args = (y, gp, priorFn)

    # Containers
    res = []
    objective = []

    # Loop over optimization calls
    for ii in range(nMinObjRestarts):

        # Keep minimizing until a valid solution is found
        while True:
            # Guess initial value from prior
            theta0 = np.array(sampleFn(1)).reshape(1,-1)

            tmp = minimize(fn, theta0, args=args, bounds=None,
                           method="nelder-mead",
                           options={"adaptive" : True})["x"]

            # If solution is finite and allowed by the prior, save!
            if np.all(np.isfinite(tmp)):
                if np.isfinite(priorFn(tmp)):
                    # Save solution, function value
                    res.append(tmp)
                    objective.append(fn(tmp, *args))
                    break

    # Return value that minimizes objective function
    return np.array(res)[np.argmin(objective)]
# end function
