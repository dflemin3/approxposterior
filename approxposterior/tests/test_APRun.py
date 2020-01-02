#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test loading approxposterior and running the core algorithm for 1 iteration.

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import approx, likelihood as lh, gpUtils
import numpy as np
import george
import emcee

def testRun():
    """
    Test the core approxposterior algorithm for several iterations until convergence.
    """

    # Define algorithm parameters
    m0 = 50                           # Initial size of training set
    m = 20                            # Number of new points to find each iteration
    nmax = 5                          # Maximum number of iterations
    bounds = [(-5,5), (-5,5)]         # Prior bounds
    algorithm = "bape"                # Use the Kandasamy et al. (2015) formalism
    seed = 57                         # For reproducibility
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

    # Create the the default GP which uses an ExpSquaredKernel
    gp = gpUtils.defaultGP(theta, y)

    # Initialize object using the Wang & Li (2017) Rosenbrock function example
    # Use default GP initialization: ExpSquaredKernel
    ap = approx.ApproxPosterior(theta=theta,
                                y=y,
                                gp=gp,
                                lnprior=lh.rosenbrockLnprior,
                                lnlike=lh.rosenbrockLnlike,
                                priorSample=lh.rosenbrockSample,
                                bounds=bounds,
                                algorithm=algorithm)

    # Run!
    ap.run(m=m, nmax=nmax, estBurnin=True, nGPRestarts=3, mcmcKwargs=mcmcKwargs,
           cache=False, samplerKwargs=samplerKwargs, verbose=False,
           thinChains=False, onlyLastMCMC=False, kmax=2, eps=1,
           convergenceCheck=True)

    # Ensure medians of chains are consistent with the true values
    samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True,
                                   thin=ap.ithins[-1])
    means = np.mean(samples, axis=0)
    trueMeans = np.array([0.0, 1.31])
    trueStds = np.array([1.5, 1.75])
    zScore = np.fabs((means - trueMeans)/trueStds)

    # Relative zScore must be close enough
    errMsg = "Medians of marginal posteriors are incosistent with true values."
    assert np.all(zScore < 1), errMsg

# end function

if __name__ == "__main__":
    testRun()
