Overview
========

A common task in science or data analysis is to perform Bayesian inference to derive a posterior probability distribution
for model parameters conditioned on some observed data with uncertainties.  In astronomy, for example, it is common
to fit for the stellar and planetary radii from observations of stellar fluxes as a function of time using the Mandel & Agol (2002)
transit model.  Typically, one can derive posterior distributions for model parameters using Markov Chain Monte Carlo (MCMC) techniques where
each MCMC iteration, one computes the likelihood of the data given the model parameters.  One must run the forward model to make
predictions to be compared against the observations and their uncertainties to compute the likelihood.  MCMC chains can require anywhere
from 10,000 to over 1,000,000 likelihood evaluations, depending on the complexity of the model and the dimensionality of the problem.
When one uses a slow forward model, one that takes minutes to run, running an MCMC analysis quickly becomes very computationally expensive.
In this case, approximate techniques are requires to compute Bayesian posterior distributions in a reasonable amount of time by minimizing
the number of forward model evaluations.

:py:obj:`approxposterior` is a Python implementation of `Bayesian Active Learning for Posterior Estimation` (BAPE_)
by Kandasamy et al. (2015) and `Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions` (AGP_) by Wang & Li (2017).
These algorithms allow the user to compute approximate posterior probability distributions using computationally expensive forward models
by training a Gaussian Process (GP) to be a surrogate for the likelihood evaluation.  The algorithms leverage the inherent uncertainty in the GP's
predictions to identify high-likelihood regions in parameter space where the GP is uncertain.  The algorithms then run the forward model at
these points to compute their likelihood and re-trains the GP to maximize the GP's predictive ability while minimizing the number of forward
model evaluations.  Check out [Bayesian Active Learning for Posterior Estimation](https://www.cs.cmu.edu/~kkandasa/pubs/kandasamyIJCAI15activePostEst.pdf) by Kandasamy et al. (2015)
and [Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions](https://arxiv.org/abs/1703.09930) by Wang & Li (2017)
for in-depth descriptions of the respective algorithms.

.. _BAPE: https://www.cs.cmu.edu/~kkandasa/pubs/kandasamyIJCAI15activePostEst.pdf
.. _AGP: https://arxiv.org/abs/1703.09930
