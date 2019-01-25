#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
docz

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import approx, gpUtils as gpu, likelihood as lh
import numpy as np
import george
import emcee, corner

seed = 15
np.random.seed(seed)

# Load in data
data = np.load("apFModelCache.npz")
theta = data["theta"]
y = data["y"]

def fitGP(theta, y, p0, seed):
    """
    Helper function to fit GP, compute lnlike
    """

    # Guess initial metric, or scale length of the covariances in loglikelihood space
    initialMetric = [1, 1, 1, 1, 1]

    # Create kernel: We'll model coverianges in loglikelihood space using a
    # Squared Expoential Kernel as we anticipate Gaussian posterior distributions
    kernel = np.log(np.var(y))*george.kernels.ExpSquaredKernel(initialMetric, ndim=theta.shape[-1])

    # Guess initial mean function
    mean = np.mean(y)

    # Create GP and compute the kernel
    gp = george.GP(kernel=kernel, fit_mean=True, mean=mean, fit_white_noise=True, white_noise=np.log(np.var(y)))
    gp.compute(theta)

    print(gp.get_parameter_names())
    print(gp.get_parameter_vector())

    print("Initial lnlike:", gp.log_likelihood(y))
    gp = gpu.optimizeGP(gp, theta, y, seed=seed, nRestarts=5, p0=gp.get_parameter_vector(), method="powell", options={})
    print("Final lnlike:", gp.log_likelihood(y))
    print("Final p:", gp.get_parameter_vector())

    return gp.log_likelihood(y), gp.get_parameter_vector()
# end function

# Containers
lnlikes = list()
p0s = list()

# Optimize the gp!
#guesses = [np.hstack(([np.mean(y)], [5.0*len(theta)**(-1.0/theta.shape[-1]) for _ in range(theta.shape[-1])])),
#         np.hstack(([np.mean(y)], np.mean(theta**2, axis=0)/theta.shape[-1]**3)),
#         [np.mean(y), 1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [np.mean(y), 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1]]

ll, p = fitGP(theta, y, p0=None, seed=seed)
print(ll, p)

# Run MCMC with best guess
# Generate initial conditions for walkers in ball around MLE solution

# emcee MCMC parameters
samplerKwargs = {"nwalkers" : 100}        # emcee.EnsembleSampler parameters
mcmcKwargs = {"iterations" : int(5.0e3)} # emcee.EnsembleSampler.run_mcmc parameters

### Initialize GP ###

# Create GP and compute the kernel
# Guess initial metric, or scale length of the covariances in loglikelihood space
initialMetric = [1, 1, 1, 1, 1]

# Create kernel: We'll model coverianges in loglikelihood space using a
# Squared Expoential Kernel as we anticipate Gaussian posterior distributions
kernel = np.log(np.var(y))*george.kernels.ExpSquaredKernel(initialMetric, ndim=theta.shape[-1])

# Guess initial mean function
mean = np.mean(y)

# Create GP and compute the kernel
gp = george.GP(kernel=kernel, fit_mean=True, mean=mean, fit_white_noise=True, white_noise=np.log(np.var(y)))
gp.set_parameter_vector(p)
gp.compute(theta)

# Initialize object using the Wang & Li (2017) Rosenbrock function example
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=lh.rosenbrockLnprior,
                            lnlike=lh.rosenbrockLnlike,
                            priorSample=lh.rosenbrockSample,
                            algorithm="bape")

sampler, iburn, ithin =  ap.runMCMC(samplerKwargs=samplerKwargs,
                                    mcmcKwargs=mcmcKwargs,
                                    cache=False, estBurnin=True,
                                    thinChains=True, verbose=True)

# Check out the final posterior distribution!

# Load in chain from last iteration
print(iburn, ithin)
if iburn < 1000:
    iburn = 2500
    ithin = 500
samples = sampler.get_chain(discard=iburn, flat=True, thin=ithin)

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True)

fig.savefig("finalPosterior.png", bbox_inches="tight") # Uncomment to save
