---
title: 'approxposterior: Approximate Posterior Distributions Python'
tags:
  - Python
authors:
  - name: David P. Fleming
    orcid: 0000-0001-9293-4043
    affiliation: University of Washington
date: 24 April 2018
bibliography: paper.bib
---

# Summary

This package is a Python implementation of "Bayesian Active Learning for Posterior Estimation" by [@Kandasamy2015] and "Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions" [@Wang2017]. These algorithms allows the user to compute approximate posterior probability distributions using computationally expensive forward models by training a Gaussian Process (GP) surrogate for the likelihood evaluation.  The algorithms leverage the inherent uncertainty in the GP's predictions to identify high-likelihood regions in parameter space where the GP is uncertain.  The algorithms then run the forward model at these points to compute their likelihood and re-trains the GP to maximize the GP's predictive ability while minimizing the number of forward model evaluations.  Check out [@Kandasamy2015] and [@Wang2017] for in-depth descriptions of the respective algorithms. *approxposterior* is under active development on GitHub and community participation is encouraged.  The code is available here [@approxposterior_github].

The following is a simple demonstration of *approxposterior* produced using an example Jupyter Notebook provided with the code on GitHub:

-![Left: Joint posterior probability distribution of the two model parameters from the Wang and Li (2017) example. The black density map denotes the true distribution while the red contours denote the approximate distribution derived using *approxposterior*. The two distributions are in excellent agreement. Right: Total computational time required to compute the posterior probability distribution of the model parameters from the Wang and Li (2017) example as a function of forward model evaluation time. The MCMC method (blue) runs the forward model for each MCMC iteration, while the orange curve was derived using the *approxposterior* BAPE implementation.](acc_scal.png)

# References
