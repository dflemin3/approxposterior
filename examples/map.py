#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script for maximum a posteriori (MAP) estimation using the Rosenbrock
function.

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
import numpy as np

# Define algorithm parameters
m0 = 50                           # Size of training set
bounds = ((-5,5), (-5,5))         # Prior bounds
algorithm = "bape"                # Use the Kandasamy et al. (2015) formalism
seed = 57                         # RNG seed
np.random.seed(seed)

# Sample design points from prior
theta = lh.rosenbrockSample(m0)

# Evaluate forward model log likelihood + lnprior for each point
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

# Initialize default gp with an ExpSquaredKernel
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

# Optimize the GP hyperparameters
ap.optGP(seed=seed, method="powell", nGPRestarts=1)

# Find MAP solution and function value at MAP
MAP, val = ap.findMAP(nRestarts=1)

# Plot MAP solution on top of grid of Rosenbrock function evaluations
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7,6))

# Generate grid of function values the old fashioned way because this function
# is not vectorized...
arr = np.linspace(-5, 5, 100)
rosen = np.zeros((100,100))
for ii in range(100):
    for jj in range(100):
        rosen[ii,jj] = lh.rosenbrockLnlike([arr[ii], arr[jj]])

# Plot Rosenbrock function (rescale because it varies by several orders of magnitude)
ax.imshow(-np.log(-rosen).T, origin="lower", aspect="auto", interpolation="nearest",
          extent=[-5, 5, -5, 5], zorder=0)

# Plot truth
ax.axhline(1, lw=2, ls=":", color="white", zorder=1)
ax.axvline(1, lw=2, ls=":", color="white", zorder=1)

# Plot MAP solution
ax.scatter(MAP[0], MAP[1], color="red", s=50, zorder=2)

# Format figure
ax.set_xlabel("x0", fontsize=15)
ax.set_ylabel("x1", fontsize=15)
title = "MAP estimate, value: (%0.3lf, %0.3lf), %e\n" % (MAP[0], MAP[1], val)
title += "Global minimum coords, value: (%0.1lf, %0.1lf), %d\n" % (1.0, 1.0, 0)
ax.set_title(title, fontsize=12)

# Save figure
fig.savefig("rosenbrockMAP.png", bbox_inches="tight")
