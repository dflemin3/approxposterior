"""

Utility functions

August 2017

@author: David P. Fleming [University of Washington, Seattle]
@email: dflemin3 (at) uw (dot) edu

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# Tell module what it's allowed to import
__all__ = ["logsubexp","AGP_utility","BAPE_utility","minimize_objective",
           "optimize_gp"]

import numpy as np
from scipy.optimize import minimize, basinhopping


################################################################################
#
# Define utility functions
#
################################################################################


def logsubexp(x1, x2):
    """
    More numerically stable way to take the log of exp(x1) - exp(x2)
    via:

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


def AGP_utility(theta, y, gp):
    """
    AGP (Adaptive Gaussian Process) utility function, the entropy of the
    posterior distribution. This is what you maximize to find the next x under
    the AGP formalism. Note here we use the negative of the utility function so
    minimizing this is the same as maximizing the actual utility function.

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object

    Returns
    -------
    u : float
        utility of theta under the gp
    """

    # Only works if the GP object has been computed, otherwise you messed up
    if gp.computed:
        mu, var = gp.predict(y, theta.reshape(1,-1), return_var=True)
    else:
        raise RuntimeError("ERROR: Need to compute GP before using it!")

    return -(mu + 1.0/np.log(2.0*np.pi*np.e*var))
    #return -(mu + np.log(2.0*np.pi*np.e*var))
# end function


def BAPE_utility(theta, y, gp):
    """
    BAPE (Bayesian Active Posterior Estimation) utility function.  This is what
    you maximize to find the next theta under the BAPE formalism.  Note here we
    use the negative of the utility function so minimizing this is the same as
    maximizing the actual utility function.  Also, we log the BAPE utility
    function as the log is monotonic so the minima are equivalent.

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object

    Returns
    -------
    u : float
        utility of theta under the gp
    """

    # Only works if the GP object has been computed, otherwise you messed up
    if gp.computed:
        mu, var = gp.predict(y, theta.reshape(1,-1), return_var=True)
    else:
        raise RuntimeError("ERROR: Need to compute GP before using it!")

    return -((2.0*mu + var) + logsubexp(var, 0.0))
# end function


def minimize_objective(fn, y, gp, sample_fn=None, prior_fn=None,
                       sim_annealing=False, **kw):
    """
    Find point that minimizes fn for a gaussian process gp conditioned on y,
    the data.

    Parameters
    ----------
    fn : function
        function to minimize that expects x, y, gp as arguments aka fn looks
        like fn_name(x, y, gp).  See *_utility functions above for examples.
    y : array
        y values to condition the gp prediction on.
    gp : george GP object
    sample_fn : function (optional)
        Function to sample initial conditions from.  Defaults to None, so we'd
        use rosenbrock_sample
    prior_fn : function (optional)
        Function to apply prior to.  If sample is rejected by prior, reject
        sample and try again.
    sim_annealing : bool (optional)
        Whether to use the simulated annealing (basinhopping) algorithm.
        Defaults to False.
    kw : dict (optional)
        Any additional keyword arguments scipy.optimize.minimize could use,
        e.g., method.

    Returns
    -------
    theta : (1 x n_dims)
        point that minimizes fn
    """

    # Assign sampling, prior function if it's not provided
    if sample_fn is None:
        sample_fn = rosenbrock_sample

    # Assign prior function if it's not provided
    if prior_fn is None:
        prior_fn = log_rosenbrock_prior

    is_finite = False
    while not is_finite:
        # Solve for theta that maximize fn and is allowed by prior

        # Choose theta0 by uniformly sampling over parameter space
        theta0 = sample_fn(1).reshape(1,-1)

        args=(y, gp)

        bounds = ((-5,5), (-5,5))
        #bounds = None

        # Mimimze fn, see if prior allows solution
        try:
            if sim_annealing:
                minimizer_kwargs = {"method":"L-BFGS-B", "args" : args,
                                    "bounds" : bounds,
                                    "options" : {"ftol" : 1.0e-3}}

                def mybounds(**kwargs):
                    x = kwargs["x_new"]
                    res = bool(np.all(np.fabs(x) < 5))
                    return res

                tmp = basinhopping(fn, theta0, accept_test=mybounds, niter=500,
                             stepsize=0.01, minimizer_kwargs=minimizer_kwargs,
                             interval=10)["x"]
            else:
                tmp = minimize(fn, theta0, args=args, bounds=bounds,
                               method="l-bfgs-b", options={"ftol" : 1.0e-3},
                               **kw)["x"]

        # ValueError.  Try again.
        except ValueError:
            tmp = np.array([np.inf for ii in range(theta0.shape[-1])]).reshape(theta0.shape)
        if np.isfinite(prior_fn(tmp).all()) and not np.isinf(tmp).any() and not np.isnan(tmp).any() and np.isfinite(tmp.sum()):
            theta = tmp
            is_finite = True
    # end while

    return np.array(theta).reshape(1,-1)
# end function


# Define the objective function (negative log-likelihood in this case).
def _nll(p, gp, y):
    """
    DOCS
    """
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25
# end function


# And the gradient of the objective function.
def _grad_nll(p, gp, y):
    """
    DOCS
    """
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)
# end function


def optimize_gp(gp, y):
    """
    DOCS

    Optimize hyperparameters of pre-computed gp
    """

    # Run the optimization routine.
    p0 = gp.get_parameter_vector()
    results = minimize(_nll, p0, jac=_grad_nll, args=(gp, y), method="bfgs")

    # Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)
    gp.recompute()
# end function
