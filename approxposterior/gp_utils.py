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
    return -gp.grad_log_likelihood(y, quiet=True)
# end function


def optimize_gp(gp, theta, y, cv=None, seed=None,
                which_kernel="ExpSquaredKernel", hyperparameters=None,
                test_size=0.25):
    """
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
    cv : int (optional)
        If not None, cv is the number (k) of k-folds CV to use.  Defaults to
        None (no CV)
    seed : int (optional)
        numpy RNG seed.  Defaults to None.
    which_kernel : str (optional)
        Name of the george kernel you want to use.  Defaults to ExpSquaredKernel
    hyperparameters : dict (optional)
        Grid of hyperparameters ranges to search over for cross-validation.
        Defaults to None.  If supplied, it should look something like this:
        {'kernel:metric:log_M_0_0': np.linspace(0.01*gp.get_parameter_vector()[0],
                                                100.0*gp.get_parameter_vector()[0],
                                                10)}
    test_size : float (optional)
        Fraction of y to use as holdout set for cross-validation.  Defaults to
        0.25.  Must be in the range (0,1).

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

        # XXX hack hack hack: this will fail when fitting for means
        hyperparameters = {'kernel:metric:log_M_0_0': np.linspace(0.01, 100.0,
                           10),
                           'kernel:metric:log_M_1_1': np.linspace(0.01, 100.0,
                                              10)}

        # Why CV if no grid given?
        if hyperparameters is None:
            err_msg = "ERROR: Trying CV but no dict of hyperparameters range given!"
            raise RuntimeError(err_msg)

        # Make a nice list of parameters
        grid = list(ParameterGrid(hyperparameters))

        # Do cv fold cross-validation
        splitter = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=seed)

        nll = []
        # Loop over each param combination
        for ii in range(len(grid)):

            iter_nll = 0.0
            for train_split, test_split in splitter.split(y):

                # Init up GP with the right dimensions
                opt_gp = setup_gp(theta[train_split], y[train_split],
                                  which_kernel="ExpSquaredKernel")

                # Set GP parameters based on current iteration
                for key in grid[ii].keys():
                    opt_gp.set_parameter(key, grid[ii][key])
                opt_gp.recompute(theta[train_split])

                # Compute NLL
                ll = opt_gp.log_likelihood(y[train_split], quiet=True)
                if np.isfinite(ll):
                    iter_nll += -ll
                else:
                    iter_nll += 1e25
            # End of iteration: append mean nll
            nll.append(iter_nll/cv)

        min_nll = np.argmin(nll)

        # Set GP parameters
        for key in grid[min_nll].keys():
            gp.set_parameter(key, grid[min_nll][key])

        # Recompute with the optimized hyperparameters!
        gp.recompute(theta)

    return gp
# end function


def setup_gp(theta, y, which_kernel="ExpSquaredKernel", mean=None, seed=None):
    """
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
        Defaults to None.  If none, estimates the mean
    seed : int (optional)
        numpy RNG seed.  Defaults to None.

    Returns
    -------
    gp : george.GP
    """

    # Guess the bandwidth
    bandwidth = np.mean(np.array(theta)**2, axis=0)/10.0

    # Which kernel?
    if str(which_kernel).lower() == "expsquaredkernel":
        kernel = george.kernels.ExpSquaredKernel(bandwidth,
                                                 ndim=np.array(theta).shape[-1])
    elif str(which_kernel).lower() == "expkernel":
        kernel = george.kernels.ExpKernel(bandwidth,
                                          ndim=np.array(theta).shape[-1])
    elif str(which_kernel).lower() == "matern32kernel":
        kernel = george.kernels.Matern32Kernel(bandwidth,
                                          ndim=np.array(theta).shape[-1])
    elif str(which_kernel).lower() == "matern52kernel":
        kernel = george.kernels.Matern52Kernel(bandwidth,
                                          ndim=np.array(theta).shape[-1])
    else:
        avail = "Available kernels: ExpSquaredKernel, ExpKernel, Matern32Kernel, Matern52Kernel"
        raise NotImplementedError("Error: Available kernels: %s" % avail)

    # Guess the mean value if nothing is given
    if mean is None:
        mean = np.mean(np.array(y), axis=0)

    # Create the GP conditioned on theta
    gp = george.GP(kernel=kernel, fit_mean=True, mean=mean)
    gp.compute(theta)

    return gp
# end function
