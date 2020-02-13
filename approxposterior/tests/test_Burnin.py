#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test estimating burn-in time of an MCMC chain

@author: David P. Fleming [University of Washington, Seattle], 2020
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import mcmcUtils
import numpy as np
import emcee


def testBurnin():
    """
    Test integrated autocorrelation length, and hence burn-in time, estimation.
    """

    np.random.seed(42)

    # Simulate simple MCMC chain based fitting a line
    # Choose the "true" parameters.
    mTrue = -0.9594
    bTrue = 4.294

    # Generate some synthetic data from the model.
    N = 50
    x = np.sort(10*np.random.rand(N))
    obserr = 0.5 # Amplitude of noise term
    obs = mTrue * x + bTrue # True model
    obs += obserr * np.random.randn(N) # Add some random noise

    # Define the loglikelihood function
    def logLikelihood(theta, x, obs, obserr):

        # Model parameters
        theta = np.array(theta)
        m, b = theta

        # Model predictions given parameters
        model = m * x + b

        # Likelihood of data given model parameters
        return -0.5*np.sum((obs-model)**2/obserr**2)


    # Define the logprior function
    def logPrior(theta):

        # Model parameters
        theta = np.array(theta)
        m, b = theta

        # Probability of model parameters: flat prior
        if -5.0 < m < 0.5 and 0.0 < b < 10.0:
            return 0.0
        return -np.inf


    # Define logprobability function: l(D|theta) * p(theta)
    # Note: use this for emcee, not approxposterior!
    def logProbability(theta, x, obs, obserr):

        lp = logPrior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + logLikelihood(theta, x, obs, obserr)

    # Randomly initialize walkers
    p0 = np.random.randn(32, 2)
    nwalkers, ndim = p0.shape

    # Set up MCMC sample object - give it the logprobability function
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logProbability, args=(x, obs, obserr))

    # Run the MCMC for 5000 iteratios
    sampler.run_mcmc(p0, 5000);

    # Estimate burnin, thin lengths
    iburn, ithin = mcmcUtils.estimateBurnin(sampler, estBurnin=True,
                                            thinChains=True, verbose=False)
    test = [iburn, ithin]

    # Compare estimated burnin to the known value
    errMsg = "burn-in, thin lengths are incorrect."
    truth = [67, 15]
    assert np.allclose(truth, test, rtol=1.0e-1), errMsg
# end function


if __name__ == "__main__":
    testBurnin()
