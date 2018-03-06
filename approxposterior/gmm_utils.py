# -*- coding: utf-8 -*-
"""

Gaussian mixture model utility functions.

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# Tell module what it's allowed to import
__all__ = ["fit_gmm"]

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture


def fit_gmm(sampler, iburn, max_comp=6, cov_type="full", use_bic=True):
    """
    Fit a Gaussian Mixture Model to the posterior samples to derive an
    approximation of the posterior density.  Fit for the number of components
    by either minimizing the Bayesian Information Criterior (BIC) or via
    cross-validation.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        sampler object containing the MCMC chains
    iburn : int
        number of burn-in steps to discard for fitting
    max_comp : int (optional)
        Maximum number of mixture model components to fit for.  Defaults to 6.
    cov_type : str (optional)
        GMM covariance type.  Defaults to "full".  See the documentation here:
        http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        for more info
    use_bic : bool (optional)
        Minimize the BIC to pick the number of GMM components or use cross
        validation?  Defaults to True (aka, use the BIC)

    Returns
    -------
    GMM : sklearn.mixture.GaussianMixture
        fitted Gaussian mixture model
    """

    # Select optimal number of components via minimizing BIC
    if use_bic:

        bic = []
        lowest_bic = 1.0e10
        best_gmm = None
        gmm = GaussianMixture()

        for n_components in range(1,max_comp+1):
          gmm.set_params(**{"n_components" : n_components,
                         "covariance_type" : cov_type})
          gmm.fit(sampler.flatchain[iburn:])
          bic.append(gmm.bic(sampler.flatchain[iburn:]))

          if bic[-1] < lowest_bic:
              lowest_bic = bic[-1]
              best_gmm = gmm

        # Refit GMM with the lowest bic
        GMM = best_gmm
        GMM.fit(sampler.flatchain[iburn:])

    # Select optimal number of components via 5 fold cross-validation
    else:
        hyperparams = {"n_components" : np.arange(max_comp+1)}
        gmm = GridSearchCV(GaussianMixture(covariance_type=cov_type),
                         hyperparams, cv=5)
        gmm.fit(sampler.flatchain[iburn:])
        GMM = gmm.best_estimator_
        GMM.fit(sampler.flatchain[iburn:])

    return GMM
# end function
