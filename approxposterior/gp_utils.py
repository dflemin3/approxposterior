"""

gaussian process functions

August 2017

@author: David P. Fleming [University of Washington, Seattle]
@email: dflemin3 (at) uw (dot) edu

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# Tell module what it's allowed to import
__all__ = ["setup_gp","optimize_gp"]

from . import bp
import numpy as np
import george
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize, basinhopping

# Define the objective function (negative log-likelihood in this case).
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


# And the gradient of the objective function.
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
    return -gp.grad_log_likelihood(y, quiet=True)
# end function


def optimize_gp(gp, y, cv=None, seed=None, which_kernel="ExpSquaredKernel"):
    """
    Optimize hyperparameters of an arbitrary george Gaussian Process kenerl
    using either a straight-up maximizing the log-likelihood or k-fold cv in which
    the log-likelihood is maximized for each fold and the best one is chosen.

    Parameters
    ----------
    gp : george.GP
    y : array
        data to condition GP on
    cv : int (optional)
        If not None, cv is the number (k) of k-folds CV to use.  Defaults to
        None (no CV)
    seed : int (optional)
        numpy RNG seed.  Defaults to None.
    which_kernel : str (optional)
        Name of the george kernel you want to use.  Defaults to ExpSquaredKernel

    Returns
    -------
    optimized_gp : george.GP
    """

    # Optimize GP by maximizing log-likelihood
    if cv is None:

        # Run the optimization routine.
        p0 = gp.get_parameter_vector()
        results = minimize(_nll, p0, jac=_grad_nll, args=(gp, y), method="bfgs")

        # Update the kernel
        gp.set_parameter_vector(results.x)
        gp.recompute()

    # Optimize GP via cv=k fold cross-validation
    else:
        # Make a bunch of folds
        splitter = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=seed)

        ii = 0 # Counter
        nll = [] # Holds negative loglikelihoods, what we want to minimize
        # Loop over said folds
        for train_split, test_split in splitter.split(y):

            # Set up temp GP with the right dimensions conditioned on this fold
            opt_gp = gp_utils.setup_gp(theta[test_split], y[test_split],
                                       which_kernel=which_kernel)

            # Set GP parameters
            for key in grid_list[ii].keys():
                opt_gp.set_parameter(key, grid_list[ii][key])

            # Compute NLL for this parameter conditioned on test data
            ll = opt_gp.log_likelihood(y[test_split], quiet=True)
            if np.isfinite(ll):
                nll.append(-ll)
            else:
                nll.append(1e25)

            ii += 1

        min_nll = np.argmin(nll)
        print(min_nll, grid_list[min_nll], nll[min_nll])

        # Set GP parameters
        for key in grid_list[min_nll].keys():
            opt_gp.set_parameter(key, grid_list[min_nll][key])

    return gp
# end function


def setup_gp(theta, y, which_kernel="ExpSquaredKernel", cv=None, seed=None):
    """

    DOCS

    init GP object and stuff
    """

    #Initial GP fit
    # Guess the bandwidth following Kandasamy et al. (2015)'s suggestion
    bandwidth = 5 * np.power(len(y),(-1.0/theta.shape[-1]))

    # Which kernel?
    if str(which_kernel).lower() == "expsquaredkernel":
        kernel = george.kernels.ExpSquaredKernel(bandwidth, ndim=theta.shape[-1])
    else:
        raise NotImplementedError("Error: Available kernels: ExpSquaredKernel")

    # Create the GP conditioned on {theta_n, log(L_n / p_n)}
    gp = george.GP(kernel=kernel)
    gp.compute(theta)

    # Optimize gp hyperparameters
    optimize_gp(gp, y, cv=cv, seed=seed, which_kernel=which_kernel)

    return gp
# end function
