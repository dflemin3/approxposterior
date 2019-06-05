Improving Performance
=====================

Not counting likelihood function evaluations, most of the computational time is
spent optimizing the GP hyperparameters and selecting new points in
parameter space to run the model that will best improve the GP's predictive performance.  Both of
these tasks are critical to the accuracy of :py:obj:`approxposterior` because
the GP must be able to learn an accurate representation of the model logprobability (loglikelihood + logprior)
throughout parameter space, particularly in high-likelihood regions, if it is
to predict an accurate approximate posterior distribution.

Both of these tasks involve minimizing a function: the marginal loglikelihood
when optimizing the GP hyperparameters and the BAPE or AGP utility function
when selecting new points. In low-dimensional problems, e.g. 2D problems like the
Rosenbrock function example, both optimizations are relatively easy.  In higher-dimensional
problems, the objective functions, e.g. the GP marginal loglikelihood and the
utility functions, both typically have multiple extrema, making optimization more
difficult.  In practice, we have found that restarting the minimization procedure
with initial guesses randomly distributed throughout parameter space results in good
performance, however, higher-dimensional problems often requires more restarts.

If :py:obj:`approxposterior` is running too slowly, or worse, is not able to derive
good posterior probability distributions, the algorithm can be tweaked in several ways.

Note, if you're interested in how :py:obj:`approxposterior` scales with forward
model evaluation time and how accurate it is, check out the Scaling and Accuracy
example notebook on the Tutorial page.

**Optimizing GP Hyperparameters**

By default, :py:obj:`approxposterior` optimizes the GP hyperparameters by
maximizing the marginal loglikelihood following the example in the george_
documentation using the approxposterior.gpUtils.optimizeGP function.

This function has a keyword argument, nGPRestarts, that defaults to 5 but can be decreased if
:py:obj:`approxposterior` is running to slowly or increased if :py:obj:`approxposterior`'s
accuracy is lacking. Set nGPRestarts to pick the number of times the GP
hyperparameter optimization should be restarted.

Furthermore, :py:obj:`approxposterior` reoptimizes the GP hyperparameters each time
a new point is selected and added to the training set.  In practice, this is likely
overkill for some applications, so the user can set optGPEveryN in the run method
to instead only reoptimize the GP every N new points. In practice, I find that
optGPEveryN = 25 usually works, but the performance can vary.

.. _george: https://george.readthedocs.io/en/latest/tutorials/hyper/

**Optimizing selecting new points in parameter space**

Similar to the GP hyperparameter optimization, finding where next in parameter space
to run the model to improve the GP's performance requires maximizing the BAPE or
AGP utility function, or minimizing its negative as we do in :py:obj:`approxposterior`.
In practice, we find that high-dimensional applications require restarting this optimization
several times, with initial guesses drawn from the prior, to achieve good performance.
As before, the keyword argument nMinObjRestarts controls the number of optimization
restarts, with the default value being 5.
