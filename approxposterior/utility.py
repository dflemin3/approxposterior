# -*- coding: utf-8 -*-
"""
:py:mod:`utility.py` - Utility Functions
----------------------------------------

Utility functions in terms of usefulness, e.g. minimizing GP utility functions
or computing KL divergences, and the GP utility functions, e.g. the bape utility.
"""

# Tell module what it's allowed to import
__all__ = ["logsubexp", "AGPUtility", "BAPEUtility", "JonesUtility",
           "minimizeObjective", "klNumerical"]

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


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
        util = -(mu + 0.5*np.log(2.0*np.pi*np.e*var))
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


def JonesUtility(theta, y, gp, priorFn, zeta=0.01):
    """
    Jones utility function - Expected Improvement derived in Jones et al. (1998)
    EI(x) = E(max(f(theta) - f(thetaBest),0)) where f(thetaBest) is the best
    value of the function so far and thetaBest is the best design point

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object
    priorFn : function
        Function that computes lnPrior probability for a given theta.
    zeta : float (optional)
        Exploration parameter. Larger zeta leads to more exploration. Defaults
        to 0.01

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
        std = np.sqrt(var)

        # Find best value
        yBest = np.max(y)

        # Intermediate quantity
        if std > 0:
            z = (mu - yBest - zeta) / std
        else:
            return 0.0

        # Standard normal CDF of z
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)

        util = -((mu - yBest - zeta) * cdf + std * pdf)
    except ValueError:
        print("Invalid util value.  Negative variance or inf mu?")
        raise ValueError("util: %e. mu: %e. var: %e" % (util, mu, var))

    return util
# end function


def minimizeObjective(fn, y, gp, sampleFn, priorFn, nRestarts=5,
                      method="nelder-mead", options=None, bounds=None,
                      theta0=None, args=None, maxIters=100):
    """
    Minimize some arbitrary function, fn. This function is most useful when
    evaluating fn requires a Gaussian process model, gp. For example, this
    function can be used to find the point that minimizes a utility fn for a gp
    conditioned on y, the data, and is allowed by the prior, priorFn.

    PriorFn is required as it helps to select against points with non-finite
    likelihoods, e.g. NaNs or infs.  This is required as the GP can only train
    on finite values.

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
    method : str (optional)
        scipy.optimize.minimize method.  Defaults to nelder-mead.
    options : dict (optional)
        kwargs for the scipy.optimize.minimize function.  Defaults to None,
        but if method == "nelder-mead", options = {"adaptive" : True}
    theta0 : float/iterable (optional)
        Initial guess for optimization. Defaults to None, which draws a sample
        from the prior function using sampleFn.
    args : iterable (optional)
        Arguments for user-specified function that this function will minimize.
        Defaults to None.
    maxIters (int) (optional)
        Maximum number of iterations to try restarting optimization if the
        solution isn't finite and/nor allowed by the prior function. Defaults to
        100.

    Returns
    -------
    thetaBest : (1 x n_dims)
        point that minimizes fn
    fnBest : float
        fn(thetaBest)
    """

    # Initialize options
    if str(method).lower() == "nelder-mead" and options is None:
        options = {"adaptive" : True}

    # Minimize GP nll, save result, evaluate marginal likelihood
    if str(method).lower() in [" l-bfgs-b", "tnc"]:
        pass
    # Bounds not allowed
    else:
        bounds = None

    if args is None:
        args = ()

    # Ensure theta0 is in the proper form, determine its dimensionality
    if theta0 is not None:
        theta0 = np.asarray(theta0).squeeze()
        ndim = theta0.ndim
        if ndim <= 0:
            ndim = 1

    # Containers
    res = []
    objective = []

    # Loop over optimization calls
    for ii in range(nRestarts):

        # Guess initial value from prior
        if theta0 is None:
            t0 = np.asarray(sampleFn(1)).reshape(1,-1)
        else:
            t0 = theta0 + np.min(theta0) * 1.0e-3 * np.random.randn(ndim)

        # Keep minimizing until a valid solution is found
        ii = 0
        while True:

            # Too many iterations
            if ii >= maxIters:
                errMsg = "ERROR: Cannot find a valid solution. Current iterations: %d\n" % ii
                errMsg += "Maximum iterations: %d\n" % maxIters
                raise RuntimeError(errMsg)

            # Minimize the function
            tmp = minimize(fn, t0, args=args, bounds=bounds,
                           method=method, options=options)["x"]

            # If solution is finite and allowed by the prior, save
            if np.all(np.isfinite(tmp)):
                if np.isfinite(priorFn(tmp)):
                    # Save solution, function value
                    res.append(tmp)
                    objective.append(fn(tmp, *args))
                    break

            # If we're here, the solution didn't work. Try again with a new
            # sample from the prior
            t0 = np.array(sampleFn(1)).reshape(1,-1)

            ii += 1
        # end loop

    # Return value that minimizes objective function out of all minimizations
    bestInd = np.argmin(objective)
    return np.array(res)[bestInd], objective[bestInd]
# end function
