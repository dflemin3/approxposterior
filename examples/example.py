"""

Example script

@author: David P. Fleming [University of Washington, Seattle]
@email: dflemin3 (at) uw (dot) edu

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


from approxposterior import bp, utility as ut, likelihood as lh
import numpy as np
import george
from george import kernels


# Define algorithm parameters
m0 = 20 # Initialize size of training set
m = 10 # Number of new points to find each iteration
nmax = 10 # Maximum number of iterations
M = int(1.0e4) # Number of MCMC steps to estimate approximate posterior
Dmax = 0.1
kmax = 5
kw = {}

# Choose m0 initial design points to initialize dataset
theta = lh.bimodal_normal_sample(m0)
y = lh.bimodal_normal_lnprob(theta)

# 0) Initial GP fit
# Guess the bandwidth following Kandasamy et al. (2015)'s suggestion
bandwidth = 5 * np.power(len(y),(-1.0/theta.shape[-1]))

# Create the GP conditioned on {theta_n, log(L_n / p_n)}
kernel = np.var(y) * kernels.ExpSquaredKernel(bandwidth, ndim=theta.shape[-1])
gp = george.GP(kernel)
gp.compute(theta)

# Optimize gp hyperparameters
ut.optimize_gp(gp, y)

# Init object
bp = bp.ApproxPosterior(gp, lnprior=lh.bimodal_normal_lnprior,
                        lnlike=lh.bimodal_normal_lnlike,
                        lnprob = lh.bimodal_normal_lnprob,
                        prior_sample=lh.bimodal_normal_sample,
                        algorithm="bape")
"""
bp = bp.ApproxPosterior(gp, lnprior=lh.rosenbrock_lnprior,
                        lnlike=lh.rosenbrock_lnlike,
                        lnprob = lh.rosenbrock_lnprob,
                        prior_sample=lh.rosenbrock_sample,
                        algorithm="bape")
"""

# Run!
bp.run(theta, y, m=m, M=M, nmax=nmax, Dmax=Dmax, kmax=kmax,
       sampler=None, sim_annealing=False, **kw)
