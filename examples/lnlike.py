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

seed = 42
np.random.seed(seed)

# Load in data
data = np.load("apFModelCache.npz")
theta = data["theta"]
y = data["y"]

print(theta.shape)
np.mean(y)

def fitGP(theta, y, p0, seed):
    """
    Helper function to fit GP, compute lnlike
    """

    # Guess initial metric, or scale length of the covariances in loglikelihood space
    initialMetric = np.mean(theta**2, axis=0)/theta.shape[-1]**3

    # Create kernel: We'll model coverianges in loglikelihood space using a
    # Squared Expoential Kernel as we anticipate Gaussian posterior distributions
    kernel = george.kernels.ExpSquaredKernel(initialMetric, ndim=theta.shape[-1])

    # Guess initial mean function
    mean = np.mean(y)

    # Create GP and compute the kernel
    gp = george.GP(kernel=kernel, fit_mean=True, mean=mean)
    gp.compute(theta)

    print("Initial lnlike:", gp.log_likelihood(y))
    gp = gpu.optimizeGP(gp, theta, y, seed=seed, nRestarts=1, p0=p0, method="basinhopping")
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
guesses = [np.array([np.mean(y), 1, 1, 1, 1, 1, 1, 1, 1, 1])]


for p0 in guesses:
    print(p0)
    p0s.append(p0)
    ll, p = fitGP(theta, y, p0, seed)
    lnlikes.append(ll)
    print()

lnlikes = np.array(lnlikes)
p0s = np.array(p0s)

# What's the best?
print(lnlikes)

bestInd = np.argmax(lnlikes)
print("Best lnlike:", lnlikes[bestInd])
print("Best guess:", p0s[bestInd])

# Run MCMC with best guess
# Generate initial conditions for walkers in ball around MLE solution
mle = np.array([1.08, 1.07, 7.28, 7.47, -0.17, -0.39, 7.27, 0.28, 2.49])
x0 = mle + 1.0e-2*np.random.randn(200, 9)

# emcee MCMC parameters
samplerKwargs = {"nwalkers" : 200}        # emcee.EnsembleSampler parameters
mcmcKwargs = {"iterations" : int(3.0e4), "initial_state" : x0} # emcee.EnsembleSampler.run_mcmc parameters

### Initialize GP ###

# Guess initial metric, or scale length of the covariances in loglikelihood space
initialMetric = p0s[bestInd][1:]

# Create kernel: We'll model coverianges in loglikelihood space using a
# Squared Expoential Kernel as we anticipate Gaussian-ish posterior
# distributions in our 2-dimensional parameter space
kernel = george.kernels.ExpSquaredKernel(initialMetric, ndim=theta.shape[-1])

# Guess initial mean function
mean = np.mean(y)

# Create GP and compute the kernel
gp = george.GP(kernel=kernel, fit_mean=True, mean=mean)
gp.compute(theta)

# Optimize GP
gp = gpu.optimizeGP(gp, theta, y, seed=seed)

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
iburn = 2500
ithin = 100
samples = sampler.get_chain(discard=iburn, flat=True, thin=ithin)

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True)

fig.savefig("finalPosterior.png", bbox_inches="tight") # Uncomment to save
