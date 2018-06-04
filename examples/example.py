#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

from approxposterior import bp, likelihood as lh
import george

### Define algorithm parameters ###
m0 = 20                           # Initial size of training set
m = 10                            # Number of new points to find each iteration
nmax = 10                         # Maximum number of iterations
M = int(1.0e2)                    # Number of MCMC steps to estimate approximate posterior
Dmax = 0.1                        # KL-Divergence convergence limit
kmax = 5                          # Number of iterations for Dmax convergence to kick in
bounds = ((-5,5), (-5,5))         # Prior bounds
algorithm = "bape"                 # Use the Kandasamy et al. (2015) formalism

### Create a training set (if you don't already have one!) ###

# Randomly sample initial conditions from the prior
theta = np.array(lh.rosenbrock_sample(m0))

# Evaluate forward model log likelihood + lnprior for each theta
y = list()
for ii in range(len(theta)):
    y.append(lh.rosenbrock_lnlike(theta[ii], *args, **kwargs) + lh.rosenbrock_lnprior(theta[ii]))
y = np.array(y)

### Initialize GP ###

# Guess initial metric
initial_metric = np.nanmedian(theta**2, axis=0)/10.0

# Create kernel
kernel = george.kernels.ExpSquaredKernel(initial_metric, ndim=2)

# Guess initial mean function
mean = np.nanmedian(y)

# Create GP
gp = george.GP(kernel=kernel, fit_mean=True, mean=mean)
gp.compute(theta)

# Initialize object using the Wang & Li (2017) Rosenbrock function example
ap = bp.ApproxPosterior(gp=gp,
                        lnprior=lh.rosenbrock_lnprior,
                        lnlike=lh.rosenbrock_lnlike,
                        prior_sample=lh.rosenbrock_sample,
                        algorithm=algorithm)

# Run!
ap.run(m0=m0, m=m, M=M, nmax=nmax, Dmax=Dmax, kmax=kmax,
       sampler=None, bounds=bounds, n_kl_samples=100000,
       verbose=True)
