.. approxposterior documentation master file, created by
   sphinx-quickstart on Thu Feb 22 12:07:36 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

approxposterior
===============

.. _Python: http://www.python.org/

:py:obj:`approxposterior` is a Python package for efficient approximate Bayesian
inference and optimization of computationally-expensive models. :py:obj:`approxposterior`
trains a Gaussian process (GP) surrogate for the computationally-expensive model
and employs an active learning approach to iteratively improve the GPs predictive
performance while minimizing the number of calls to the expensive model required
to generate the GP's training set.

:py:obj:`approxposterior` implements variants of `Bayesian Active Learning for Posterior Estimation` (BAPE_)
by Kandasamy et al. (2015) and `Adaptive Gaussian process approximation for
Bayesian inference with expensive likelihood functions` (AGP_) by Wang & Li (2017).
These active learning algorithms outline schemes for GP active learning that :py:obj:`approxposterior`
uses for its Bayesian inference and/or optimization.

For information on how to install :py:obj:`approxposterior`, numerous example use cases, and detailed API
documentation, check out the Table of Contents to the left and below.

Introduction
============

:py:obj:`approxposterior` is a Python implementation of both the `Bayesian Active Learning for Posterior Estimation` (BAPE_) and `Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions` (AGP_) algorithms for estimating posterior probability distributions for use with inference problems with computationally-expensive models. In such situations,
the goal is to infer posterior probability distributions for model parameters, given some data, with the additional constraint of minimizing the number of forward model evaluations given the model's assumed large computational cost.  :py:obj:`approxposterior` trains a Gaussian Process (GP) surrogate model for the likelihood evaluation by modeling the covariances in logprobability (logprior + loglikelihood) space. :py:obj:`approxposterior` then uses this GP within an MCMC sampler for each likelihood evaluation to perform the inference. :py:obj:`approxposterior` iteratively improves the GP's predictive performance by leveraging the inherent uncertainty in the GP's predictions to identify high-likelihood regions in parameter space where the GP is uncertain.  :py:obj:`approxposterior` then evaluates the forward model at these points to expand the training set in relevant regions of parameter space, re-training the GP to maximize its predictive ability while minimizing the size of the training set.  Check out (AGP_) and (BAPE_) for in-depth descriptions of the respective algorithms.

In practice, we find that :py:obj:`approxposterior` can estimate posterior probability distributions that are accurate
approximations to the true, underlying distributions with only of order 100s-1000s model evaluations to train the GP, compared to 1,000,000, often more, required by MCMC methods, depending on the inference problem. The estimated marginal posterior distributions have medians that are all typically within a few percent of the true values, with similar uncertainties to the true distributions.  We have validated :py:obj:`approxposterior` for 2-5 dimensional problems, while Kandasamy et al. (2015) found in an 9-dimensional case that the BAPE algorithm significantly outperformed MCMC methods in terms of both accuracy and speed. See their paper for details and check out the examples for more information and example use cases.

.. _BAPE: https://www.cs.cmu.edu/~kkandasa/pubs/kandasamyIJCAI15activePostEst.pdf
.. _AGP: https://arxiv.org/abs/1703.09930

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   map
   Fitting a Line <notebooks/fittingALine.ipynb>
   Scaling and Accuracy <notebooks/ScalingAccuracy.ipynb>
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
