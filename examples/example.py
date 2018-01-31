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

# Define algorithm parameters
m0 = 20                           # Initial size of training set
m = 10                            # Number of new points to find each iteration
nmax = 10                         # Maximum number of iterations
M = int(1.0e4)                    # Number of MCMC steps to estimate approximate posterior
Dmax = 0.1                        # KL-Divergence convergence limit
kmax = 5                          # Number of iterations for Dmax convergence to kick in
which_kernel = "ExpSquaredKernel" # Which Gaussian Process kernel to use
bounds = ((-5,5), (-5,5))         # Prior bounds
algorithm = "bape"                 # Use the Kandasamy et al. (2015) formalism

# Initialize object using the Wang & Li (2017) Rosenbrock function example
ap = bp.ApproxPosterior(lnprior=lh.rosenbrock_lnprior,
                        lnlike=lh.rosenbrock_lnlike,
                        lnprob=lh.rosenbrock_lnprob,
                        prior_sample=lh.rosenbrock_sample,
                        algorithm=algorithm)

# Run!
ap.run(m0=m0, m=m, M=M, nmax=nmax, Dmax=Dmax, kmax=kmax, cv=cv,
        sampler=None, bounds=bounds, which_kernel=which_kernel,
        n_kl_samples=100000, verbose=True, debug=True)
