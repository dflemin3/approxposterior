Tutorial
========

Check out the example notebooks to see how the BAPE algorithm works, how the code runtime scales
for different forward model evaluation times, how we compute the Kullback–Leibler (KL) divergence,
how we can use :py:obj:`approxposterior`
and how we compute the true posterior distribution for the Rosenbrock function example from Wang & Li (2017).

.. toctree::
   :maxdepth: 1
   :caption: Jupyter Notebook Examples:

   Example <notebooks/example.ipynb>
   Fitting a Line <notebooks/fittingALine.ipynb>
   Scaling and Accuracy <notebooks/ScalingAccuracy.ipynb>
   KL Divergence Estimation <notebooks/KLDivergenceEstimation.ipynb>
   Posterior Fitting with Gaussian Mixture Models <notebooks/posteriorFittingWithGMM.ipynb>
   Rosenbrock Function Example <notebooks/TrueRosenbrockPosterior.ipynb>

Below is a quick example of how to use :py:obj:`approxposterior` to compute the posterior
distribution of the Rosenbrock Function example from Wang & Li (2017) using the
BAPE algorithm. To keep track of the MCMC progress, set verbose = True in the appoxposterior.run method. This setting
outputs X/M where M is the total number of MCMC iterations to be evaluated, 5,000 in this example, and x is the current
iteration number.  Note that setting verbose = True also outputs additional diagnostic information, such as when
the MCMC finishes, what the estimated burn-in is, and other quantities that are useful for tracking the progress of
your code.  In this example, we set verbose = False for simplicity.

1) First, the user must set model parameters.

.. code-block:: python

  from approxposterior import approx, likelihood as lh
  import numpy as np

  # Define algorithm parameters
  m0 = 50                           # Initial size of training set
  m = 20                            # Number of new points to find each iteration
  nmax = 10                         # Maximum number of iterations
  Dmax = 0.1                        # KL-Divergence convergence limit
  kmax = 5                          # Number of iterations for Dmax convergence to kick in
  nKLSamples = 10000                # Number of samples from posterior to use to calculate KL-Divergence
  bounds = ((-5,5), (-5,5))         # Prior bounds
  algorithm = "BAPE"                # Use the Kandasamy et al. (2015) formalism

  # emcee MCMC parameters
  samplerKwargs = {"nwalkers" : 20}        # emcee.EnsembleSampler parameters
  mcmcKwargs = {"iterations" : int(2.0e4)} # emcee.EnsembleSampler.run_mcmc parameters

2) Create an initial training set and gaussian process

.. code-block:: python

  # Randomly sample initial conditions from the prior
  theta = np.array(lh.rosenbrockSample(m0))

  # Evaluate forward model log likelihood + lnprior for each theta
  y = np.zeros(len(theta))
  for ii in range(len(theta)):
      y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

  # Create the the default GP which uses an ExpSquaredKernel
  gp = gpUtils.defaultGP(theta, y)

3) Initialize the :py:obj:`approxposterior` object.

.. code-block:: python

  # Initialize object using the Wang & Li (2017) Rosenbrock function example
  ap = approx.ApproxPosterior(theta=theta,
                              y=y,
                              gp=gp,
                              lnprior=lh.rosenbrockLnprior,
                              lnlike=lh.rosenbrockLnlike,
                              priorSample=lh.rosenbrockSample,
                              algorithm=algorithm)

4) Run!

.. code-block:: python

  ap.run(m=m, nmax=nmax, Dmax=Dmax, kmax=kmax, bounds=bounds,  estBurnin=True,
         nKLSamples=nKLSamples, mcmcKwargs=mcmcKwargs, cache=False,
         samplerKwargs=samplerKwargs, verbose=True)

5) Examine the final posterior distributions!

.. code-block:: python

  # Check out the final posterior distribution!
  import corner

  # Load in chain from last iteration
  samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

  # Corner plot!
  fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                      scale_hist=True, plot_contours=True)

  #fig.savefig("finalPosterior.png", bbox_inches="tight") # Uncomment to save

The final posterior distribution will look something like the following:

.. image:: _figures/final_posterior.png
  :width: 400
  :alt: Final posterior distribution for approxposterior run of the Wang & Li (2017) example.
