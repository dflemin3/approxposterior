"""

Example script

@author: David P. Fleming [University of Washington, Seattle]
@email: dflemin3 (at) uw (dot) edu

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


from approxposterior import bp, likelihood as lh
import numpy as np


# Define algorithm parameters
m0 = 20 # Initialize size of training set
m = 10 # Number of new points to find each iteration
nmax = 10 # Maximum number of iterations
M = int(5e3) # Number of MCMC steps to estimate approximate posterior
Dmax = 0.1
kmax = 5
cv = None #10
kw = {}
bounds = ((-5,5), (-5,5))

# Init object
bp = bp.ApproxPosterior(lnprior=lh.rosenbrock_lnprior,
                        lnlike=lh.rosenbrock_lnlike,
                        lnprob = lh.rosenbrock_lnprob,
                        prior_sample=lh.rosenbrock_sample,
                        algorithm="bape", bounds=bounds)

# Run!
bp.run(m0=m0, m=m, M=M, nmax=nmax, Dmax=Dmax, kmax=kmax, cv=cv,
       sampler=None, sim_annealing=False, **kw)
