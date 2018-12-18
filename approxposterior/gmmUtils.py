# -*- coding: utf-8 -*-
"""

Gaussian mixture model utility functions.

"""

# Tell module what it's allowed to import
__all__ = ["fitGMM"]

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture


def fitGMM(samples, maxComp=3, covType="full", useBic=True):
    """
    Fit a Gaussian Mixture Model to the posterior samples to derive an
    approximation of the posterior density.  Fit for the number of components
    by either minimizing the Bayesian Information Criterior (BIC) or via
    cross-validation.

    Parameters
    ----------
    samples : numpy array
        sampler.flatchain MCMC chain array of dimensions (nwalkers x nsteps)
    maxComp : int (optional)
        Maximum number of mixture model components to fit for.  Defaults to 3.
    covType : str (optional)
        GMM covariance type.  Defaults to "full".  See the documentation here:
        http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        for more info
    useBic : bool (optional)
        Minimize the BIC to pick the number of GMM components or use cross
        validation?  Defaults to True (aka, use the BIC)

    Returns
    -------
    GMM : sklearn.mixture.GaussianMixture
        fitted Gaussian mixture model
    """

    # Select optimal number of components via minimizing BIC
    if useBic:

        bic = None
        lowestBic = np.inf
        bestGMM = None
        gmm = GaussianMixture()

        for nComponents in range(1,maxComp+1):
          gmm.set_params(**{"n_components" : nComponents,
                         "covariance_type" : covType})
          gmm.fit(samples)
          bic = gmm.bic(samples)

          if bic < lowestBic:
              lowestBic = bic
              bestN = nComponents
              bestCovType = covType

        # Refit GMM with the lowest bic
        GMM = GaussianMixture(n_components=bestN, covariance_type=bestCovType)
        GMM.fit(samples)

    # Select optimal number of components via 5 fold cross-validation
    else:
        hyperparams = {"n_components" : np.arange(maxComp+1)}
        gmm = GridSearchCV(GaussianMixture(covariance_type=covType),
                         hyperparams, cv=5)
        gmm.fit(samples)
        GMM = gmm.best_estimator_
        GMM.fit(samples)

    return GMM
# end function
