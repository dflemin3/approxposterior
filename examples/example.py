#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
import numpy as np

# Define algorithm parameters
m0 = 50                           # Initial size of training set
m = 20                            # Number of new points to find each iteration
nmax = 2                          # Maximum number of iterations
bounds = ((-5,5), (-5,5))         # Prior bounds
algorithm = "bape"                # Use the Kandasamy et al. (2015) formalism
seed = 57                         # RNG seed

np.random.seed(seed)

# emcee MCMC parameters
samplerKwargs = {"nwalkers" : 20}        # emcee.EnsembleSampler parameters
mcmcKwargs = {"iterations" : int(2.0e4)} # emcee.EnsembleSampler.run_mcmc parameters

# Sample initial conditions from prior
theta = lh.rosenbrockSample(m0)

# Evaluate forward model log likelihood + lnprior for each theta
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

# Default gp
gp = gpUtils.defaultGP(theta, y, white_noise=-10)

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
ap.run(m=m, nmax=nmax, estBurnin=True, nGPRestarts=1, mcmcKwargs=mcmcKwargs,
       cache=False, samplerKwargs=samplerKwargs, verbose=True, thinChains=True,
       onlyLastMCMC=True)

# Check out the final posterior distribution!
import corner

# Load in chain from last iteration
samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True)

# Plot where forward model was evaluated - uncomment to plot!
fig.axes[2].scatter(ap.theta[m0:,0], ap.theta[m0:,1], s=10, color="red", zorder=20)

# Save figure
fig.savefig("finalPosterior.png", bbox_inches="tight")
