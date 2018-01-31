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
m0 = 20 # Initialize size of training set
m = 10 # Number of new points to find each iteration
nmax = 10 # Maximum number of iterations
M = int(1.0e4) # Number of MCMC steps to estimate approximate posterior
Dmax = 0.1
kmax = 5
cv = None
which_kernel = "ExpSquaredKernel"
bounds = ((-5,5), (-5,5))

# Init object
bape = bp.ApproxPosterior(lnprior=lh.rosenbrock_lnprior,
                         lnlike=lh.rosenbrock_lnlike,
                         lnprob = lh.rosenbrock_lnprob,
                         prior_sample=lh.rosenbrock_sample,
                         algorithm="bape")

# Run!
bape.run(m0=m0, m=m, M=M, nmax=nmax, Dmax=Dmax, kmax=kmax, cv=cv,
        sampler=None, bounds=bounds, which_kernel=which_kernel,
        n_kl_samples=100000, verbose=True, debug=True)
