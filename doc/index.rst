.. approxposterior documentation master file, created by
   sphinx-quickstart on Thu Feb 22 12:07:36 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

approxposterior
===============

A Python implementation of `Bayesian Active Learning for Posterior Estimation
<https://www.cs.cmu.edu/~kkandasa/pubs/kandasamyIJCAI15activePostEst.pdf/>` _.
by Kandasamy et al. (2015) and [Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions](https://arxiv.org/abs/1703.09930)
by Wang & Li (2017).


Installation
============
Clone the repository and run:

``python setup.py install``

.. ipython::

  In [1]: import numpy as np

  In [2]: x = np.arange(10)

  In [3]: x**2
  Out[3]: array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

  In [4]: import matplotlib.pyplot as plt

  @savefig plot_simple.png width=4in
  In [5]: plt.plot(x);

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

The main class is :class:`~approxposterior.ApproxPosterior`.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
