.. approxposterior documentation master file, created by
   sphinx-quickstart on Thu Feb 22 12:07:36 2018.

approxposterior
===============

:py:obj:`approxposterior` is a Python package for efficient approximate Bayesian
inference and Bayesian optimization of computationally-expensive models. :py:obj:`approxposterior`
trains a Gaussian process (GP) surrogate model for the computationally-expensive model
and employs an active learning approach to iteratively improve the GPs predictive
performance while minimizing the number of calls to the expensive model required
to generate the GP's training set.

:py:obj:`approxposterior` implements variants of `Bayesian Active Learning for Posterior Estimation` (BAPE_)
by Kandasamy et al. (2017) and `Adaptive Gaussian process approximation for
Bayesian inference with expensive likelihood functions` (AGP_) by Wang & Li (2018).
These active learning algorithms outline schemes for GP active learning that :py:obj:`approxposterior`
uses for its Bayesian posterior inference. :py:obj:`approxposterior` implemented the Jones et al. (1998)
`Expected Utility` function for Bayesian optimization.

For information on how to install :py:obj:`approxposterior`, numerous examples, and detailed API
documentation, check out the Table of Contents to the left and below.

.. _BAPE: http://www.sciencedirect.com/science/article/pii/S0004370216301394
.. _AGP: https://www.semanticscholar.org/paper/Adaptive-Gaussian-Process-Approximation-for-with-Wang-Li/a11e3a4144898920835ccff7ef0ed0b159b94bc6

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   bayesopt
   map
   Fitting a Line <notebooks/fittingALine.ipynb>
   Scaling and Accuracy <notebooks/ScalingAccuracy.ipynb>
   api
   install
   faq
   citation
   Github <https://github.com/dflemin3/approxposterior>
   Submit an Issue <https://github.com/dflemin3/approxposterior/issues>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
