Tutorial
========

Check out the example notebooks to see how the BAPE algorithm works, how the code runtime scales
for different forward model evaluation times, how we compute the Kullback–Leibler (KL) divergence,
how we can use :py:obj:`approxposterior`
and how we compute the true posterior distribution for the Rosenbrock function example from Wang & Li (2017).

.. toctree::
   :maxdepth: 1
   :caption: Jupyter Notebook Examples:

   BAPE Example <notebooks/BAPE_Example.ipynb>
   Scaling <notebooks/Scaling_Accuracy.ipynb>
   KL Divergence Example <notebooks/KL_Divergence_Estimation.ipynb>
   Forecast Example <notebooks/Forecast_Example.ipynb>
   Rosenbrock Example <notebooks/True_Rosenbrock_Posterior.ipynb>

Below is a quick example of how to use :py:obj:`approxposterior` to compute the posterior
distribution of the Rosenbrock Function example from Wang & Li (2017) using the
BAPE algorithm. Note that although this example is relatively straight-forward,
it is still computationally non-trivial and will take of order 10 minutes to run.

1) First, the user must set model parameters.

.. code-block:: python

  from approxposterior import bp, likelihood as lh

  # Define algorithm parameters
  m0 = 200                          # Initial size of training set
  m = 20                            # Number of new points to find each iteration
  nmax = 2                          # Maximum number of iterations
  M = int(5.0e3)                    # Number of MCMC steps to estimate approximate posterior
  Dmax = 0.1                        # KL-Divergence convergence limit
  kmax = 5                          # Number of iterations for Dmax convergence to kick in
  which_kernel = "ExpSquaredKernel" # Which Gaussian Process kernel to use
  bounds = ((-5,5), (-5,5))         # Prior bounds
  algorithm = "bape"                # Use the Kandasamy et al. (2015) formalism

2) Initialize the :py:obj:`approxposterior` object.

.. code-block:: python

  # Initialize object using the Wang & Li (2017) Rosenbrock function example
  ap = bp.ApproxPosterior(lnprior=lh.rosenbrock_lnprior,
                          lnlike=lh.rosenbrock_lnlike,
                          prior_sample=lh.rosenbrock_sample,
                          algorithm=algorithm)

3) Run!

.. code-block:: python

  # Run!
  ap.run(m0=m0, m=m, M=M, nmax=nmax, Dmax=Dmax, kmax=kmax,
          sampler=None, bounds=bounds, which_kernel=which_kernel,
          n_kl_samples=100000, verbose=False)

4) Examine the final posterior distributions!

.. code-block:: python

  # Import corner to examine posterior distributions
  import corner

  fig = corner.corner(ap.samplers[-1].flatchain[ap.iburns[-1]:],
                      quantiles=[0.16, 0.5, 0.84], show_titles=True, scale_hist=True,
                      plot_contours=True)

  #fig.savefig("final_dist.png", bbox_inches="tight") # Uncomment to save

The final posterior distribution will look something like the following:

.. image:: _figures/final_posterior.png
  :width: 400
  :alt: Final posterior distribution for approxposterior run of the Wang & Li (2017) example.
