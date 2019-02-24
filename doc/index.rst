.. approxposterior documentation master file, created by
   sphinx-quickstart on Thu Feb 22 12:07:36 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

approxposterior
===============

.. _Python: http://www.python.org/

A Python implementation of `Bayesian Active Learning for Posterior Estimation` (BAPE_)
by Kandasamy et al. (2015) and `Adaptive Gaussian process approximation for
Bayesian inference with expensive likelihood functions` (AGP_) by Wang & Li (2017). For information
on how to install :py:obj:`approxposterior`, numerous example use cases, and detailed API
documentation, check out the Table of Contents to the left and below.

Motivation
==========

Given a set of observations, often one wishes to infer model parameters given the data. To do so, one
can use Bayesian inference to derive a posterior probability distribution
for model parameters conditioned on the observed data, with uncertainties.  In astronomy, for example, it is common
to fit for transiting planetary radii from observations of stellar fluxes as a function of time using the Mandel & Agol (2002)
transit model.  Typically, one can derive posterior distributions for model parameters using Markov Chain Monte Carlo (MCMC) techniques where
each MCMC iteration, one computes the likelihood of the data given the model parameters.   To compute the likelihood,
one evaluates the model to make predictions to be compared against the observations, e.g. with a Chi^2 metric.  In practice, MCMC analyses can require anywhere
from 10,000 to over 1,000,000 likelihood evaluations, depending on the complexity of the model and the dimensionality of the problem.
When using a slow model, e.g. one that takes minutes to run, running an MCMC analysis quickly becomes very computationally expensive, and often intractable.
In this case, approximate techniques are required to compute Bayesian posterior distributions in a reasonable amount of time by minimizing
the number of model evaluations.

:py:obj:`approxposterior` is a Python implementation of `Bayesian Active Learning for Posterior Estimation` (BAPE_)
by Kandasamy et al. (2015) and `Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions` (AGP_) by Wang & Li (2017).
These algorithms allow the user to compute full approximate posterior probability distributions for inference
problems with computationally expensive models.  These algorithms work by training a Gaussian Process (GP) to
effectively become a surrogate model for the likelihood evaluation by modeling the covariances in
logprobability space. To improve the GP's own predictive performance, both algorithms leverage the inherent
uncertainty in the GP's predictions to identify high-likelihood regions in parameter space where the GP is uncertain.
The algorithms then evaluate the model at these points to compute the likelihood and re-trains the GP to maximize
the GP's predictive ability while minimizing the number of model evaluations.  Check out BAPE_ by Kandasamy et al. (2015)
and AGP_ by Wang & Li (2017) for in-depth descriptions of the respective algorithms.

In practice, we find that :py:obj:`approxposterior` can derive full approximate joint posterior probability distributions that are accurate
approximations to the true, underlying distributions with only of order 100s-1000s model evaluations to train the GP,
compared to 100,000, often more, required by MCMC methods, depending on the inference problem.
The approximate marginal posterior distributions have medians that are all typically within 1-5%
of the true values, with similar uncertainties to the true distributions.  We have validated
:py:obj:`approxposterior` for 2-5 dimensional problems, while Kandasamy et al. (2015) found in an 9-dimensional
case that the BAPE algorithm significantly outperformed MCMC methods in terms of accuracy and speed.
See their paper for details and check out the examples for more information and example use cases.

.. _BAPE: https://www.cs.cmu.edu/~kkandasa/pubs/kandasamyIJCAI15activePostEst.pdf
.. _AGP: https://arxiv.org/abs/1703.09930

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   tutorial
   api
   install
   performance
   citation
   Github <https://github.com/dflemin3/approxposterior>
   Submit an Issue <https://github.com/dflemin3/approxposterior/issues>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
