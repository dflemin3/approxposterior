#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test loading approxposterior and running the core algorithm for 1 iteration.

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import approx, likelihood as lh
import numpy as np
import george
import emcee

def test_run():
    """
    Test the core approxposterior algorithm for 2 iterations.
    """

    # Define algorithm parameters
    m0 = 200                          # Initial size of training set
    m = 20                            # Number of new points to find each iteration
    nmax = 2                          # Maximum number of iterations
    Dmax = 0.1                        # KL-Divergence convergence limit
    kmax = 5                          # Number of iterations for Dmax convergence to kick in
    bounds = ((-5,5), (-5,5))         # Prior bounds
    algorithm = "bape"                # Use the Kandasamy et al. (2015) formalism
    seed = 42                         # For reproducibility
    np.random.seed(seed)

    # emcee MCMC parameters
    mcmcKwargs = {"iterations" : int(5.0e3)} # Number of MCMC steps
    samplerKwargs = {"nwalkers" : 20}        # emcee.EnsembleSampler parameters

    # Randomly sample initial conditions from the prior
    theta = np.array(lh.rosenbrockSample(m0))

    # Evaluate forward model log likelihood + lnprior for each theta
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

    # Initialize object using the Wang & Li (2017) Rosenbrock function example
    # Use default GP initialization: ExpSquaredKernel
    ap = approx.ApproxPosterior(theta=theta,
                                y=y,
                                lnprior=lh.rosenbrockLnprior,
                                lnlike=lh.rosenbrockLnlike,
                                priorSample=lh.rosenbrockSample,
                                algorithm=algorithm)

    # Run!
    ap.run(m=m, nmax=nmax, Dmax=Dmax, kmax=kmax, bounds=bounds, cache=False,
           nKLSamples=100000, mcmcKwargs=mcmcKwargs, samplerKwargs=samplerKwargs,
           verbose=False, nGPRestarts=3, seed=seed)

    # Ensure medians of chains are consistent with the true values
    samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])
    x1Med, x2Med = np.median(samples, axis=0)

    diffX1 = np.fabs(0.04 - x1Med)
    diffX2 = np.fabs(1.29 - x2Med)

    # Differences between estimated and true medians must be close-ish, but not
    # perfect because we've using a small number of samples to make this test
    # quick enough
    errMsg = "Medians of marginal posteriors are incosistent with true values."
    assert((diffX1 < 0.5) & (diffX2 < 0.5)), errMsg

# end function

if __name__ == "__main__":
    test_run()
