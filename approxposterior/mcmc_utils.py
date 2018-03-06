# -*- coding: utf-8 -*-
"""

MCMC utility functions.

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# Tell module what it's allowed to import
__all__ = ["autocorr","estimate_burnin"]

import numpy as np
import emcee
from scipy.interpolate import UnivariateSpline


def autocorr(x):
    """
    Compute the autocorrelation function

    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation

    Parameters
    ----------
    x : array

    Returns
    -------
    result : array
        ACF
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result
# end function


def estimate_burnin(sampler, nwalk, nsteps, ndim):
    """
    Given an MCMC chain, estimate the burn-in time (credit: Jacob Lustig-Jaeger)
    This function computes the maximum autocorrelation length of all the walkers
    that clearly haven't strayed too far from the converged answer.  If your
    chains have converged, this function provides a conservative estimate of the
    burn-in.  As with all things, MCMC, your mileage will vary.  Currently this
    function just supports emcee.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
    nwalk : int
        Number of walkers
    nsteps : int
        Number of MCMC steps (iterations)
    ndim : int
        Data dimensionality (number of parameters)

    Returns
    -------
    iburn : int
        Index corresponding to estimated burn-in length scale

    """

    iburn = 0
    ikeep = []
    autoc = []
    autolength = []

    walkers = np.arange(nwalk)
    iterations = np.arange(nsteps)

    # Loop over number of free parameters
    for j in range(ndim):

        # Loop over walkers
        for i in range(nwalk):
            # Get list of other walker indicies
            walkers = np.arange(nwalk)
            other_iwalkers = np.delete(walkers, i)

            # Calculate the median of this chain
            med_chain =  np.median(sampler.chain[i,iburn:,j])

            # Calculate the mean of this chain
            mean_chain =  np.mean(sampler.chain[i,iburn:,j])

            # Calculate the median and std of all the other chains
            med_other = np.median(sampler.chain[other_iwalkers,iburn:,j])
            std_other = np.std(sampler.chain[other_iwalkers,iburn:,j])

            # If this chain is within 3-sig from all other chain's median
            if np.fabs(mean_chain - med_other) < 3*std_other:
                # Keep it!
                ikeep.append(i)

                # Get autocorrelation of chain
                autoci = autocorr(sampler.chain[i,iburn:,j])
                autoc.append(autoci)

                # Fit with spline
                spline = UnivariateSpline(iterations, autoci, s=0)

                # Find zero crossings
                roots = spline.roots()

                # Save autocorrelation length
                autolength.append(np.min(roots))

    # List of chains that we are keeping
    ikeep = list(set(ikeep))

    # Set burn-in index to maximum autocorrelation length
    return int(np.max(autolength))
# end function
