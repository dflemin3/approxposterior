Bayesian Optimization
=====================

:py:obj:`approxposterior` can be used to find an accurate approximation of the
maximum (or minimum) of a function using Bayesian optimization. :py:obj:`approxposterior`
initially trains on a small number of function evaluations and then maximizes
a utility function, here Jones et al. (1998)'s "Expected Utility", to identify .
This method is particularly useful when the function in question is
computationally-expensive to evaluate so one wishes to minimizes the number of evaluations.
For some great reviews on the theory of Bayesian Optimization and it's implementation,
check out: `this great blog post by Martin Krasser <https://krasserm.github.io/2018/03/21/bayesian-optimization/>`_
and this excellent review paper by `Brochu et al. (2010) <https://arxiv.org/abs/1012.2599>`_

Below is an example of how to use :py:obj:`approxposterior` to estimate the
maximum of a 1D function using both Bayesian Optimization and maximum a posteriori
(MAP) estimation using the function learned by GP surrogate model.

Note that in general, :py:obj:`approxposterior` wants to maximize functions since
it is designed for approximate probabilistic inference (e.g., we are interested in
maximum likelihood solutions), so keep this in mind when coding up your own
objective functions. If you're instead interested in minimizing some function,
throw a `-` sign in front of your function.

For this example, we wish to find the maximum of the following objective function:

.. code-block:: python

  # Plot objective function
  fig, ax = plt.subplots(figsize=(6,5))

  x = np.linspace(-1, 2, 100)
  ax.plot(x, lh.testBOFn(x), lw=2.5, color="k")

  # Format
  ax.set_xlim(-1.1, 2.1)
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel(r"f($\theta$)")

  # Hide top, right axes
  ax.spines["right"].set_visible(False)
  ax.spines["top"].set_visible(False)
  ax.yaxis.set_ticks_position("left")
  ax.xaxis.set_ticks_position("bottom")

  fig.savefig("objFn.png", bbox_inches="tight", dpi=200)

  .. image:: _figures/objFn.png
    :width: 400

This objective function has a clear maximum, but also a local maximum, so it
should be a reasonable test. Now to the optimization.

1) First, the user must set model parameters.

.. code-block:: python

  # Define algorithm parameters
  m0 = 3                           # Size of initial training set
  bounds = [[-1, 2]]               # Prior bounds
  algorithm = "jones"              # Expected Utility from Jones et al. (1998)
  numNewPoints = 10                # Maximum number of new design points to find
  seed = 91                        # RNG seed
  np.random.seed(seed)

2) Create an initial training set and Gaussian process

.. code-block:: python

  # Sample design points from prior to create initial training set
  theta = lh.testBOFnSample(m0)

  # Evaluate forward model + lnprior for each point
  y = np.zeros(len(theta))
  for ii in range(len(theta)):
      y[ii] = lh.testBOFn(theta[ii]) + lh.testBOFnLnPrior(theta[ii])

  # Initialize default gp with an ExpSquaredKernel
  gp = gpUtils.defaultGP(theta, y, white_noise=-12, fitAmp=True)

3) Initialize the :py:obj:`approxposterior` object

.. code-block:: python

  # Initialize object using a simple 1D test function, optimize GP hyperparameters
  ap = approx.ApproxPosterior(theta=theta,
                              y=y,
                              gp=gp,
                              lnprior=lh.testBOFnLnPrior,
                              lnlike=lh.testBOFn,
                              priorSample=lh.testBOFnSample,
                              bounds=bounds,
                              algorithm=algorithm)

  # Optimize the GP hyperparameters
  ap.optGP(seed=seed, method="powell", nGPRestarts=1)

4) Perform Bayesian Optimization

.. code-block:: python

  # Run the Bayesian optimization!
  soln = ap.bayesOpt(nmax=numNewPoints, tol=1.0e-3, seed=seed, verbose=False,
                     cache=False, gpMethod="powell", optGPEveryN=1, nGPRestarts=2,
                     nMinObjRestarts=5, initGPOpt=True, minObjMethod="nelder-mead",
                     gpHyperPrior=gpUtils.defaultHyperPrior, findMAP=True)

Note that in this step, we did several things that are worth pointing out. First,
the `soln` dictionary returned by ap.bayesOpt contains several parameters, including
the solution path, `soln[thetas]`, and the value of the function at each theta, `soln[vals]`.
`soln[thetaBest]` and `soln[valBest]` represent the coordinates and function value
as the maximum, respectively.

Second, we this Bayesian optimization until either nmax iterations were ran or
the best solution changed by <= tol = 1.0e-3. In this case, only 9 iterations
were run, so the solution converged at the specified tolerance. Additionally,
by setting optGPEveryN = 1, we re-optimized the GP hyperparameters each time ap
added a new point to its training set. Keeping optGPEveryN to low values will
tend to produce more accurate solutions as, especially for early iterations, the
GP's approximate function can change quickly as it gains more information as the
training set expands.

Third, in addition to finding the Bayesian optimization solution, we set `findMAP=True`
to have :py:obj:`approxposterior` also find the maximum a posteriori (MAP) solution.
That is, the :py:obj:`approxposterior` identified the maximum of the approximate
function learned by the GP. This optimization is rather cheap since it does not
require evaluating the forward model. Since :py:obj:`approxposterior`'s goal is
to have its GP actively learn an approximation to the objective function, its
maximum should be approximately equal to the true maximum.

Below, we'll compare the Bayesian optimization and MAP solution paths contained
in `soln`.

5) Compare :py:obj:`approxposterior` BayesOpt, MAP solution to truth:

.. code-block:: python

  import matplotlib
  import matplotlib.pyplot as plt
  matplotlib.rcParams.update({"font.size": 15})

  # Plot the solution path and function value convergence
  fig, axes = plt.subplots(ncols=2, figsize=(12,6))

  # Extract number of iterations ran by bayesopt routine
  iters = [ii for ii in range(soln["nev"])]

  # Left: solution
  axes[0].axhline(trueSoln["x"], ls="--", color="k", lw=2)
  axes[0].plot(iters, soln["thetas"], "o-", color="C0", lw=2.5, label="GP BayesOpt")
  axes[0].plot(iters, soln["thetasMAP"], "o-", color="C1", lw=2.5, label="GP approximate MAP")

  # Format
  axes[0].set_ylabel(r"$\theta$")
  axes[0].legend(loc="best", framealpha=0, fontsize=14)

  # Right: solution value (- true soln since we minimized -fn)
  axes[1].axhline(-trueSoln["fun"], ls="--", color="k", lw=2)
  axes[1].plot(iters, soln["vals"], "o-", color="C0", lw=2.5)
  axes[1].plot(iters, soln["valsMAP"], "o-", color="C1", lw=2.5)

  # Format
  axes[1].set_ylabel(r"$f(\theta)$")

  # Format both axes
  for ax in axes:
      ax.set_xlabel("Iteration")
      ax.set_xlim(-0.2, soln["nev"]-0.8)

      # Hide top, right axes
      ax.spines["right"].set_visible(False)
      ax.spines["top"].set_visible(False)
      ax.yaxis.set_ticks_position("left")
      ax.xaxis.set_ticks_position("bottom")

  fig.savefig("bo.png", dpi=200, bbox_inches="tight")

.. image:: _figures/bo.png
  :width: 400

:py:obj:`approxposterior` MAP solution: (1.021, 1.041), 3.506784e-4 (red point),
compared to the truth (1,1), 0 (white dashed lines).
Our answer is pretty close to the truth, and better yet, :py:obj:`approxposterior`
only required 50 randomly-distributed Rosenbrock function evaluations to train
its GP used to estimate the MAP solution. For computationally-expensive
forward models, this method can be used for efficient (approximate) Bayesian
optimization of functions.
