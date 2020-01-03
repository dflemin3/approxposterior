FAQs
====

**Optimizing GP Hyperparameters**

Not counting likelihood function evaluations, most of the computational time is
spent optimizing the GP hyperparameters. This task is critical to the accuracy of :py:obj:`approxposterior` because
the GP must be able to learn an accurate representation of the model logprobability (loglikelihood + logprior)
throughout parameter space, particularly in high-likelihood regions, if it is
to predict an accurate approximate posterior distribution.

To optimize the GP hyperparameters, :py:obj:`approxposterior` maximizes the marginal loglikelihood
of the GP training set. In low-dimensional problems, e.g. 2D problems like the
Rosenbrock function example, this optimization is relatively easy.  In higher-dimensional
problems, the GP marginal loglikelihood typically has multiple extrema, making optimization more
difficult.  In practice, we have found that restarting the minimization procedure
with random initial guesses results in good performance, however, higher-dimensional
problems likely require more restarts.

If :py:obj:`approxposterior` is running too slowly, or worse, is not able to estimate
good posterior probability distributions, the algorithm can be tweaked in several ways.

Note: If you're interested in how :py:obj:`approxposterior` scales with forward
model evaluation time and how accurate it is, check out the Scaling and Accuracy
example notebook on the Tutorial page.

By default, :py:obj:`approxposterior` optimizes the GP hyperparameters by
maximizing the marginal loglikelihood following the example in the george_
documentation using the approxposterior.gpUtils.optimizeGP. This function is typically
used by the approxposterior.ApproxPosterior.optGP method which is a wrapped for
the aforementioned function.

optGP has a keyword argument, nGPRestarts, that sets the number of times the GP
hyperparameter optimization is restarted. Decrease nGPRestarts if
:py:obj:`approxposterior` is running too slowly or increase it if the GP's
accuracy is lacking.

Furthermore, :py:obj:`approxposterior` reoptimizes the GP hyperparameters each time
a new point is selected and added to the training set.  In practice, this is likely
overkill as the GP usually quickly learns a reasonable approximation to the
objective function, and hence we do not expect the hyperparameters to change significantly
if the training set expands by 1. The user can set optGPEveryN in the run method
to instead only reoptimize the GP every N new points. We typically reoptimize the
hyperparameters about twice per iteration.

.. _george: https://george.readthedocs.io/en/latest/tutorials/hyper/

**Optimizing selecting new points in parameter space**

Similar to the GP hyperparameter optimization, finding where next in parameter space
to run the model to improve the GP's performance requires maximizing a utility function.
In practice, we find that high-dimensional applications require restarting this optimization
several times, with initial guesses drawn from the prior, to achieve good performance.
As before, the keyword argument nMinObjRestarts controls the number of optimization
restarts, with the default value being 5. If good points are not being selected,
try increasing this parameter.
