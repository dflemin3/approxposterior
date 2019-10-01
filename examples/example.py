#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
import numpy as np
import george

# Define algorithm parameters
m0 = 50                           # Initial size of training set
m = 20                            # Number of new points to find each iteration
nmax = 2                          # Maximum number of iterations
Dmax = 0.1                        # KL-Divergence convergence limit
kmax = 5                          # Number of iterations for Dmax convergence to kick in
nKLSamples = 10000                # Number of samples from posterior to use to calculate KL-Divergence
bounds = ((-5,5), (-5,5))         # Prior bounds
algorithm = "BAPE"                # Use the Kandasamy et al. (2015) formalism

# emcee MCMC parameters
samplerKwargs = {"nwalkers" : 20}        # emcee.EnsembleSampler parameters
mcmcKwargs = {"iterations" : int(2.0e4)} # emcee.EnsembleSampler.run_mcmc parameters

# Sample initial conditions from the prior
theta = lh.rosenbrockSample(m0)

# Evaluate forward model log likelihood + lnprior for each theta
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

# Create the the default GP which uses an ExpSquaredKernel
#gp = gpUtils.defaultGP(theta, y)

initialMetric = np.array([5.0*len(theta)**(-1.0/theta.shape[-1]) for _ in range(theta.shape[-1])])
kernel = george.kernels.ExpSquaredKernel(initialMetric, ndim=theta.shape[-1])
kernel = kernel + george.kernels.LinearKernel(log_gamma2=initialMetric[0],
                                              order=1, bounds=None,
                                              ndim=theta.shape[-1])
gp = george.GP(kernel=kernel, fit_mean=True, mean=np.mean(y),
               white_noise=1.0e-10, fit_white_noise=False)
gp.compute(theta)

p0 = initialMetric + [initialMetric[0]]

# Initialize object using the Wang & Li (2017) Rosenbrock function example
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=lh.rosenbrockLnprior,
                            lnlike=lh.rosenbrockLnlike,
                            priorSample=lh.rosenbrockSample,
                            bounds=bounds,
                            algorithm=algorithm)

# Run!
ap.run(m=m, nmax=nmax, Dmax=Dmax, kmax=kmax, estBurnin=True, nGPRestarts=1,
       nKLSamples=nKLSamples, mcmcKwargs=mcmcKwargs, cache=False, gpP0=p0,
       samplerKwargs=samplerKwargs, verbose=True, onlyLastMCMC=True)

# Check out the final posterior distribution!
import corner

# Load in chain from last iteration
samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True)

fig.savefig("finalPosterior.png", bbox_inches="tight") # Uncomment to save
