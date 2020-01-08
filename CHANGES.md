0.4
===

* Bayesian Optimization via the ap.bayesOpt method, Jones et al. (1998) Expected Utility function

* New optional convergence check for ap.run method based on the inferred posterior distributions 

* Numerous additional tests, refining exists tests to be more robust

* Many documentation improvements, updates to existing examples, new examples

* Restructured examples directory to be more intuitive

* Host of bug fixes so approxposterior can better infer dimensionality, catch and handle errors, etc

* Added more badges to the main README, including a more obvious link to the docs

0.3
===

* Added option to have default kernel be sum of ExpSquaredKernel and LinearKernel for Bayesian linear regression

* Major stability improvements to enable more robust GP kernels, including amplitude and white noise terms

* Switched default GP hyperparameter optimization algorithm to powell

* Removed convergence based on KL-divergence metrics as the numerical integration is slow and noisy in high dimensions

* Added a new utility function option: naive

* Made optGP, a function for optimizing the GP hyperparameters, a class method

* Added ability to pass GP hyperparameter optimization priors

* Added new function to find maximum a posteriori (MAP) estimate given a trained GP

* Added new example, tests for MAP functions

* Added ability to ignore initial training set after 0th iteration (sometimes useful, apparently?)

* Added many new tests for GP optimization, finding next point, etc

* Added ability to toggle whether or not to fit for the GP kernel amplitude
  (sometimes can be very numerically unstable when the amp is included for high
  dimensional cases, even with regularization)

* Many documentation improvements, removal of deprecated code

0.2.post1
=========

Same as 0.2, but with typo fixes.

0.2
===

Stable version 0.2 release

* Contains all chains in 0.2rc0, but now validated and stable.

0.2rc0
======

Version 0.2 0th release candidate

* Dramatically improved GP hyperparameter optimization, selecting new design points for higher-dimensional cases.

* Numerous bug fixes and resolved issues (see issues page)

* Now depends on emcee version 3.0 or greater

* Significantly refactored code, breaking the 0.1 API, for easy of use and generalized functions to work with a wider range of models.

* Added additional examples and extensive API documentation.

0.1.post1
=========

Bug fixes to 0.1 release to address bugs and other issues raised in JOSS review.

* See https://github.com/openjournals/joss-reviews/issues/781 for a full description and discussion of the changes.

0.1
===

Initial stable release

* Implemented basic functionality of BAPE and AGP algorithms
