***approxposterior***

A Python implementation of [Bayesian Active Learning for Posterior Estimation](https://www.cs.cmu.edu/~kkandasa/pubs/kandasamyIJCAI15activePostEst.pdf) by Kandasamy et al. (2015) and [Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions](https://arxiv.org/abs/1703.09930) by Wang & Li (2017).

[![build status](http://img.shields.io/travis/dflemin3/approxposterior/master.svg?style=flat)](https://travis-ci.org/dflemin3/approxposterior)

Installation
============
Clone the repository and run

```bash
python setup.py install
```

A simple example
===================

```python
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
algorithm = "agp"                 # Use the Wang & Li (2017) formalism

# Initialize object using the Wang & Li (2017) Rosenbrock function example
ap = bp.ApproxPosterior(lnprior=lh.rosenbrock_lnprior,
                        lnlike=lh.rosenbrock_lnlike,
                        prior_sample=lh.rosenbrock_sample,
                        algorithm=algorithm)

# Run!
ap.run(m0=m0, m=m, M=M, nmax=nmax, Dmax=Dmax, kmax=kmax,
       bounds=bounds, which_kernel=which_kernel)
```     
Please cite this repository and both Kandasamy et al. (2015) and Wang & Li (2017).
