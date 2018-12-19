# -*- coding: utf-8 -*-
"""

MCMC utility functions.

"""

# Tell module what it's allowed to import
__all__ = ["validateMCMCKwargs", "autocorr", "estimateBurnin"]

import numpy as np
import emcee
import warnings
from scipy.interpolate import UnivariateSpline

import emcee
version = emcee.__version__
assert int(version.split(".")[0]) > 2, "approxposterior is only compatible with emcee versions >= 3"


def validateMCMCKwargs(samplerKwargs, mcmcKwargs, ap, verbose=False):
    """
    Validates emcee.EnsembleSampler parameters/kwargs

    Parameters
    ----------
    samplerKwargs : dict
        dictionary containing parameters intended for emcee.EnsembleSampler
        object
    mcmcKwargs : dict
        dictionary containing parameters intended for
        emcee.EnsembleSampler.run_mcmc/.sample object
    ap : approxposterior.ApproxPosterior
    verbose : bool (optional)
        verboisty level. Defaults to False (no output)
    Returns
    -------
    samplerKwargs : dict
        Sanitized dictionary containing parameters intended for
        emcee.EnsembleSampler object
    mcmcKwargs : dict
        Sanitized dictionary containing parameters intended for
        emcee.EnsembleSampler.run_mcmc/.sample object
    """

    # First validate kwargs for emcee.EnsembleSampler object
    if samplerKwargs is None:
        samplerKwargs = dict()

        samplerKwargs["nwalkers"] = 10 * samplerKwargs["dim"]
        samplerKwargs["log_prob_fn"] = ap._gpll
    else:
        try:
            nwalkers = samplerKwargs["nwalkers"]
        except KeyError:
            print("WARNING: samplerKwargs provided but nwalkers not in samplerKwargs")
            print("Defaulting to nwalkers = 10 * dim")
            samplerKwargs["nwalkers"] = 10 * samplerKwargs["ndim"]

        # Handle case when user supplies own loglikelihood function
        if "log_prob_fn" in samplerKwargs.keys():
            if verbose:
                print("WARNING: log_prob_fn in samplerKwargs. approxposterior only uses the GP surrogate model for the lnlikelihood!")
                print("Disregarding log_prob_fn...")

            # Remove any other log_prob_fn
            samplerKwargs.pop("log_prob_fn", None)

        # Properly initialize log_prob_fn to be GP loglikelihood estimate
        samplerKwargs["log_prob_fn"] = ap._gpll

    # Validate mcmcKwargs dict used in sampling posterior distribution
    # e.g. emcee.EnsembleSampler.run_mcmc method
    if mcmcKwargs is None:
        mcmcKwargs = dict()
        mcmcKwargs["iterations"] = 10000
        mcmcKwargs["initial_state"] = emcee.State(np.asarray([ap.priorSample(1) for j in range(samplerKwargs["nwalkers"])]))
    else:
        try:
            nsteps = mcmcKwargs["iterations"]
        except KeyError:
            mcmcKwargs["iterations"] = 10000
            if verbose:
                print("WARNING: mcmcKwargs provided, but N not in mcmcKwargs.")
                print("Defaulting to N = 10000.")
        try:
            p0 = mcmcKwargs["initial_state"]
        except KeyError:
            mcmcKwargs["initial_state"] = emcee.State(np.asarray([ap.priorSample(1) for j in range(samplerKwargs["nwalkers"])]))
            if verbose:
                print("WARNING: mcmcKwargs provided, but p0 not in mcmcKwargs.")
                print("Defaulting to nwalkers initial states from priorSample.")

    print(np.shape(mcmcKwargs["initial_state"].coords))

    return samplerKwargs, mcmcKwargs
# end function



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


def estimateBurnin(sampler, nwalk, nsteps, ndim):
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

                # Find zero crossing
                roots = spline.roots()

                # Save autocorrelation length
                # If there are no roots, warn user that iburn = 1
                try:
                    min_root = np.min(roots)
                except ValueError:
                    min_root = 0
                    warn_msg = "WARNING: Burn-in estimation failed.  iburn set to 0."
                    warnings.warn(warn_msg)

                autolength.append(min_root)

    # List of chains that we are keeping
    ikeep = list(set(ikeep))

    # Set burn-in index to maximum autocorrelation length
    return int(np.max(autolength))
# end function
