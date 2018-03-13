Tutorial
========

Check out the example notebooks to see how the BAPE algorithm works, how the code runtime scales
for different forward model evaluation times, how we compute the Kullback–Leibler (KL) divergence,
and how we compute the true posterior distribution for the Rosenbrock function example from Wang & Li (2017).

.. toctree::
   :maxdepth: 1
   :caption: Jupyter Notebook Examples:

   BAPE Example <notebooks/BAPE_Example.ipynb>
   Scaling <notebooks/Scaling_Accuracy.ipynb>
   KL Divergence Example <notebooks/KL_Divergence_Estimation.ipynb>
   Rosenbrock Example <notebooks/True_Rosenbrock_Posterior.ipynb>

Below is a quick example of how to use :py:obj:`approxposterior` to compute the posterior
distribution of the Rosenbrock Function example from Wang & Li (2017) using the
BAPE algorithm.

1) First, the user must set model parameters.

.. code-block:: python

  from approxposterior import bp, likelihood as lh

  # Define algorithm parameters
  m0 = 20                           # Initial size of training set to generate (if None is supplied)
  m = 10                            # Number of new points to find each iteration
  nmax = 10                         # Maximum number of iterations
  M = int(1.0e4)                    # Number of MCMC steps to estimate approximate posterior
  Dmax = 0.1                        # KL-Divergence convergence limit
  kmax = 5                          # Number of iterations for Dmax convergence to kick in
  which_kernel = "ExpSquaredKernel" # Which Gaussian Process kernel to use
  bounds = ((-5,5), (-5,5))         # Prior bounds
  algorithm = "agp"                 # Use the Wang & Li (2017) formalism

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
         bounds=bounds, which_kernel=which_kernel)
