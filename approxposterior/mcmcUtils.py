# -*- coding: utf-8 -*-
"""
:py:mod:`mcmcUtils.py` - Markov Chain Monte Carlo Utility Functions
-----------------------------------

MCMC utility functions for validating emcee MCMC runs within approxposterior.
"""

# Tell module what it's allowed to import
__all__ = ["validateMCMCKwargs"]

import numpy as np
import emcee

import emcee
version = emcee.__version__
assert int(version.split(".")[0]) > 2, "approxposterior is only compatible with emcee versions >= 3"


def validateMCMCKwargs(ap, samplerKwargs, mcmcKwargs, verbose=False):
    """
    Validates emcee.EnsembleSampler parameters/kwargs.

    Parameters
    ----------
    ap : approxposterior.ApproxPosterior
            Initialized ApproxPosterior object
    samplerKwargs : dict
        dictionary containing parameters intended for emcee.EnsembleSampler
        object
    mcmcKwargs : dict
        dictionary containing parameters intended for
        emcee.EnsembleSampler.run_mcmc/.sample object
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

        samplerKwargs["ndim"] = ap.theta.shape[-1]
        samplerKwargs["nwalkers"] = 20 * samplerKwargs["dim"]
        samplerKwargs["log_prob_fn"] = ap._gpll
    else:
        # If user set ndim, ignore it and align it with theta's dimensionality
        samplerKwargs.pop("ndim", None)
        samplerKwargs["ndim"] = ap.theta.shape[-1]

        # Initialize other parameters if they're not provided
        try:
            nwalkers = samplerKwargs["nwalkers"]
        except KeyError:
            print("WARNING: samplerKwargs provided but nwalkers not in samplerKwargs")
            print("Defaulting to nwalkers = 20 per dimension.")
            samplerKwargs["nwalkers"] = 20 * samplerKwargs["ndim"]

        if "backend" in samplerKwargs.keys():
            print("WARNING: backend in samplerKwargs. approxposterior creates its own!")
            print("with filename = apRun.h5. Disregarding user-supplied backend.")

        # Handle case when user supplies own loglikelihood function
        if "log_prob_fn" in samplerKwargs.keys():
            if verbose:
                print("WARNING: log_prob_fn in samplerKwargs. approxposterior only uses the GP surrogate model for the lnlikelihood!")
                print("Disregarding log_prob_fn...")

            # Remove any other log_prob_fn
            samplerKwargs.pop("log_prob_fn", None)

        # Prevent users from providing their own backend
        samplerKwargs.pop("backend", None)

        # Properly initialize log_prob_fn to be GP loglikelihood estimate
        samplerKwargs["log_prob_fn"] = ap._gpll

    # Validate mcmcKwargs dict used in sampling posterior distribution
    # e.g. emcee.EnsembleSampler.run_mcmc method
    if mcmcKwargs is None:
        mcmcKwargs = dict()
        mcmcKwargs["iterations"] = 10000
        mcmcKwargs["initial_state"] = ap.priorSample(samplerKwargs["nwalkers"])
    else:
        try:
            nsteps = mcmcKwargs["iterations"]
        except KeyError:
            mcmcKwargs["iterations"] = 10000
            if verbose:
                print("WARNING: mcmcKwargs provided, but iterations not in mcmcKwargs.")
                print("Defaulting to iterations = 10000.")
        try:
            p0 = mcmcKwargs["initial_state"]
        except KeyError:
            mcmcKwargs["initial_state"] = ap.priorSample(samplerKwargs["nwalkers"])
            if verbose:
                print("WARNING: mcmcKwargs provided, but initial_state not in mcmcKwargs.")
                print("Defaulting to nwalkers samples from priorSample.")

    return samplerKwargs, mcmcKwargs
# end function
