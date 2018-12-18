#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

from approxposterior import approx, likelihood as lh
import numpy as np
import george


# Define algorithm parameters
m0 = 50                           # Initial size of training set
m = 20                            # Number of new points to find each iteration
nmax = 2                          # Maximum number of iterations
Dmax = 0.1                        # KL-Divergence convergence limit
kmax = 5                          # Number of iterations for Dmax convergence to kick in
nKLSamples = 100000               # Number of samples from posterior to use to calculate KL-Divergence
bounds = ((-5,5), (-5,5))         # Prior bounds
algorithm = "bape"                # Use the Kandasamy et al. (2015) formalism

# emcee MCMC parameters
samplerKwargs = {"nwalkers" : 20}        # emcee.EnsembleSampler parameters
mcmcKwargs = {"iterations" : int(2.0e4)} # emcee.EnsembleSampler.run_mcmc parameters

# Randomly sample initial conditions from the prior
theta = np.array(lh.rosenbrockSample(m0))

# Evaluate forward model log likelihood + lnprior for each theta
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

### Initialize GP ###

# Guess initial metric, or scale length of the covariances in loglikelihood space
initialMetric = np.nanmedian(theta**2, axis=0)/10.0

# Create kernel: We'll model coverianges in loglikelihood space using a
# Squared Expoential Kernel as we anticipate Gaussian-ish posterior
# distributions in our 2-dimensional parameter space
kernel = george.kernels.ExpSquaredKernel(initialMetric, ndim=2)

# Guess initial mean function
mean = np.nanmedian(y)

# Create GP and compute the kernel
gp = george.GP(kernel=kernel, fit_mean=True, mean=mean)
gp.compute(theta)

# Initialize object using the Wang & Li (2017) Rosenbrock function example
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=lh.rosenbrockLnprior,
                            lnlike=lh.rosenbrockLnlike,
                            priorSample=lh.rosenbrockSample,
                            algorithm=algorithm)

# Run!
ap.run(m0=m0, m=m, nmax=nmax, Dmax=Dmax, kmax=kmax, bounds=bounds,
       estBurnin=True, nKLSamples=nKLSamples, mcmcKwargs=mcmcKwargs,
       samplerKwargs=samplerKwargs, verbose=True)

# Check out the final posterior distribution!
import corner

fig = corner.corner(ap.samplers[-1].flatchain[ap.iburns[-1]:],
                            quantiles=[0.16, 0.5, 0.84], show_titles=True,
                            scale_hist=True, plot_contours=True)

fig.savefig("finalPosterior.png", bbox_inches="tight") # Uncomment to save
