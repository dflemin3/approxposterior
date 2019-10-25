# -*- coding: utf-8 -*-
"""
:py:mod:`gpUtils.py` - Gaussian Process Utilities
-----------------------------------

Gaussian process utility functions, e.g. optimizing GP hyperparameters.

"""

# Tell module what it's allowed to import
__all__ = ["optimizeGP"]

from . import utility as util
import numpy as np
import george
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


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

    # Negative gradient of log likelihood
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)
# end function


def defaultGP(theta, y, order=None, white_noise=-10):
    """
    Basic utility function that initializes a simple GP that works well in many
    applications, but is not guaranteed to work in general.

    Parameters
    ----------
    theta : array
        Design points
    y : array
        Data to condition GP on, e.g. the lnlike * lnprior at each design point,
        theta.
    order : int (optional)
        Order of PolynomialKernel to add to ExpSquaredKernel. Defaults to None,
        that is, no PolynomialKernel is added and the GP only uses the
        ExpSquaredKernel
    white_noise : float (optional)
        From george docs: "A description of the logarithm of the white noise
        variance added to the diagonal of the covariance matrix". Defaults to
        log(white_noise) = -10. Note: if order is not None, you might need to
        set the white_noise to a large value for the computation to be
        numerically stable, but this, as always, depends on the application.

    Returns
    -------
    gp : george.GP
        Gaussian process with initialized kernel and factorized covariance
        matrix.
    """

    # Guess initial metric, or scale length of the covariances in loglikelihood space
    # using suggestion from Kandasamy et al. (2015)
    initialMetric = np.array([5.0*len(theta)**(-1.0/theta.shape[-1]) for _ in range(theta.shape[-1])])

    # Create kernel: We'll model coveriances in loglikelihood space using a
    # Squared Expoential Kernel
    kernel = np.var(y) * george.kernels.ExpSquaredKernel(initialMetric,
                                                         bounds=None,
                                                         ndim=theta.shape[-1])

    # Add a linear regression kernel of order order?
    # Use a meh guess for the amplitude and for the scale length (gamma)
    if order is not None:
        kernel = kernel + (np.var(y)/10.0) * george.kernels.LinearKernel(log_gamma2=initialMetric[0],
                                                                         order=order,
                                                                         bounds=None,
                                                                         ndim=theta.shape[-1])

    # Create GP and compute the kernel, aka factor the covariance matrix
    gp = george.GP(kernel=kernel, fit_mean=False, mean=np.mean(y),
                   white_noise=white_noise, fit_white_noise=False,
                   solver=george.HODLRSolver)
    gp.compute(theta)

    return gp
# end function


def optimizeGP(gp, theta, y, seed=None, nGPRestarts=5, method=None,
               options=None, p0=None, gpCV=None):
    """
    Optimize hyperparameters of an arbitrary george Gaussian Process kernel
    by maximizing the marginalized log-likelihood.

    Parameters
    ----------
    gp : george.GP
    theta : array
    y : array
        data to condition GP on
    seed : int (optional)
        numpy RNG seed.  Defaults to None.
    nGPRestarts : int (optional)
        Number of times to restart the optimization.  Defaults to 5. Increase
        this number if the GP isn't optimized well.
    method : str (optional)
        scipy.optimize.minimize method.  Defaults to l-bfgs-b if None.
    options : dict (optional)
        kwargs for the scipy.optimize.minimize function.  Defaults to None, or
        an empty dictionary.
    p0 : array (optional)
        Initial guess for kernel hyperparameters.  If None, defaults to
        ndim values randomly sampled from a uniform distribution over [-10, 10)
    gpCV : int (optional)
        Whether or not to use k-fold cross-validation to select kernel
        hyperparameters from the nGPRestarts maximum likelihood solutions.
        This can be useful if the GP is overfitting, but will likely slow down
        the code. Defaults to None, aka this functionality is not used. If using
        it, perform gpCV-fold cross-validation.

    Returns
    -------
    optimizedGP : george.GP
    """

    # Set default parameters if None are provided
    if method is None:
        method = "l-bfgs-b"
    if options is None:
        options = dict()

    # Run the optimization routine nGPRestarts times
    res = []
    mll = []

    # Optimize GP hyperparameters by maximizing marginal log_likelihood
    for ii in range(nGPRestarts):
        # Inputs for each process
        if p0 is None:
            # Pick random guesses for kernel hyperparameters from reasonable range
            k1ConstGuess = np.random.normal(loc=np.log(np.var(y)), scale=np.sqrt(np.log(np.var(y))))
            metricGuess = [np.random.uniform(low=-10, high=10) for _ in range(theta.shape[-1])]

            # If a linear regression kernel is included, add guesses for initial parameters
            if("kernel:k2:k1:log_constant" in gp.get_parameter_names()):
                k2ConstGuess = np.random.normal(loc=np.log(np.var(y)/10.0), scale=np.sqrt(np.log(np.var(y)/10.0)))
                k2GammaGuess = np.random.normal(loc=np.log(np.var(y)/10.0), scale=np.sqrt(np.log(np.var(y)/10.0)))

                # Stack the guesses
                x0 = np.hstack(([k1ConstGuess],
                                 metricGuess,
                                [k2ConstGuess, k2GammaGuess]))
            # Just 1 kernel: stack guesses
            else:
                x0 = np.hstack(([k1ConstGuess], metricGuess))

        else:
            # Take user-supplied guess and slightly perturb it
            x0 = np.array(p0) + np.min(p0) * 1.0e-3 * np.random.randn(len(p0))

        # Minimize GP nll, save result, evaluate marginal likelihood
        resii = minimize(_nll, x0, args=(gp, y), method=method, jac=_grad_nll,
                         bounds=gp.kernel.bounds, options=options)["x"]
        res.append(resii)

        # Update the kernel with solution for computing marginal loglike
        gp.set_parameter_vector(resii)
        gp.recompute()

        # Compute marginal log likelihood for this set of kernel hyperparameters
        mll.append(gp.log_likelihood(y, quiet=True))

    # Use CV to select best answer?
    if gpCV is not None:
        if isinstance(gpCV, int):
            mses = np.zeros((gpCV, nGPRestarts))

            # Use gpCV fold cross-validation
            kfold = KFold(n_splits=gpCV)

            # Train on train, evaluate predictions on test
            ii = 0
            for trainInds, testInds in kfold.split(theta, y):
                # Repeat for each solution
                for jj in range(len(res)):
                    # Update the kernel using training set
                    gp.set_parameter_vector(res[ii])
                    gp.compute(theta[trainInds])

                    # Compute marginal log likelihood for this set of
                    # kernel hyperparameters conditioned on the training set
                    yhat = gp.predict(y[trainInds], theta[testInds],
                                      return_cov=False, return_var=False)
                    mses[ii,jj] = mean_squared_error(y[testInds], yhat)

                # End loop over each MLL solution for this cv fold
                ii = ii + 1

            # Best answer is solution with minimum mean squared error
            # averaging over the folds
            ind = np.argmin(np.mean(mses, axis=0))
        else:
            raise RuntimeError("gpCV must be an integer. gpCV:", gpCV)

    # Pick result with largest marginal log likelihood
    else:
        ind = np.argmax(mll)

    # Update gp
    gp.set_parameter_vector(res[ind])
    if gpCV is not None:
        gp.compute(theta)
    else:
        gp.recompute()

    return gp
# end function
