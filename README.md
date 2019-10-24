***approxposterior***

Overview
========

Given a set of observations, often one wishes to infer model parameters given the data. To do so, one
can use Bayesian inference to derive a posterior probability distribution
for model parameters conditioned on the observed data, with uncertainties.  In astronomy, for example, it is common
to fit for a transiting planet's radius from observations of stellar fluxes as a function of time using the Mandel & Agol (2002)
transit model.  Typically, one can derive posterior distributions for model parameters using Markov Chain Monte Carlo (MCMC) techniques where each MCMC iteration, one computes the likelihood of the data given the model parameters.   To compute the likelihood,
one evaluates the model to make predictions to be compared against the observations, e.g. with a Chi^2 metric.  In practice, MCMC analyses can require anywhere from 10,000 to over 1,000,000 likelihood evaluations, depending on the complexity of the model and the dimensionality of the problem. When using a slow model, e.g. one that takes minutes to run, running an MCMC analysis quickly becomes very computationally expensive, and is often intractable. In this case, approximate techniques are required to compute Bayesian posterior distributions in a reasonable amount of time by minimizing the number of model evaluations.

approxposterior is a Python implementation of [Bayesian Active Learning for Posterior Estimation](https://www.cs.cmu.edu/~kkandasa/pubs/kandasamyIJCAI15activePostEst.pdf)
by Kandasamy et al. (2015) and [Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions](https://arxiv.org/abs/1703.09930) by Wang & Li (2017).
These algorithms allow the user to compute full approximate posterior probability distributions for inference problems with computationally expensive models.  These algorithms work by training a Gaussian Process (GP) to effectively become a surrogate model for the likelihood evaluation by modeling the covariances in logprobability space. To improve the GP's own predictive performance, both algorithms leverage the inherent uncertainty in the GP's predictions to identify high-likelihood regions in parameter space where the GP is uncertain.  The algorithms then evaluate the model at these points to compute the likelihood and re-trains the GP to maximize the GP's predictive ability while minimizing the number of model evaluations.  Check out [Bayesian Active Learning for Posterior Estimation](https://www.cs.cmu.edu/~kkandasa/pubs/kandasamyIJCAI15activePostEst.pdf) by Kandasamy et al. (2015) and [Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions](https://arxiv.org/abs/1703.09930) by Wang & Li (2017)
for in-depth descriptions of the respective algorithms.

In practice, we find that approxposterior can derive full approximate joint posterior probability distributions that are accurate
approximations to the true, underlying distributions with only of order 100s-1000s model evaluations to train the GP, compared to 100,000, often more, required by MCMC methods, depending on the inference problem. The approximate marginal posterior distributions have medians that are all typically within 1-5% of the true values, with similar uncertainties to the true distributions.  We have validated approxposterior for 2-5 dimensional problems, while Kandasamy et al. (2015) found in an 9-dimensional case that the BAPE algorithm significantly outperformed MCMC methods in terms of accuracy and speed. See their paper for details and check out the examples for more information and example use cases.

Code Status and Documentation
=============================

approxposterior runs on Python 3.5, 3.6, and 3.7.

[![build status](http://img.shields.io/travis/dflemin3/approxposterior/master.svg?style=flat)](https://travis-ci.org/dflemin3/approxposterior)

[![DOI](http://joss.theoj.org/papers/10.21105/joss.00781/status.svg)](https://doi.org/10.21105/joss.00781)

Check out the [documentation](https://dflemin3.github.io/approxposterior/) for a more in-depth explanation about the code,
detailed API documentation, numerous examples.

Installation
============

Using pip:

```bash
pip install approxposterior
```

This step can fail if george (the Python Gaussian Process package) is not properly installed and compiled.
To install george, run

```bash
    conda install -c conda-forge george
```

From source:

```bash
git clone https://github.com/dflemin3/approxposterior.git
cd approxposterior
python setup.py install
```

A simple example
===================

Below is a simple application of approxposterior based on the Wang & Li (2017) example. Note that
we adapted this example and shortened it so that it only takes about 1 minute to run.

To keep track of the MCMC progress, set ```verbose = True``` in the ```approx.run``` method. This setting
outputs X/M where M is the total number of MCMC iterations to be evaluated, 5,000 in this example, and x is the current
iteration number.  Note that setting ```verbose = True``` also outputs additional diagnostic information, such as when
the MCMC finishes, what the estimated burn-in is, and other quantities that are useful for tracking the progress of
your code.  In this example, we set ```verbose = False``` for simplicity.

```python
from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
import numpy as np
import george

# Define algorithm parameters
m0 = 50                           # Initial size of training set
m = 20                            # Number of new points to find each iteration
nmax = 2                          # Maximum number of iterations
bounds = ((-5,5), (-5,5))         # Prior bounds
algorithm = "BAPE"                # Use the Kandasamy et al. (2015) formalism
seed = 57                         # RNG seed
np.random.seed(seed)

# emcee MCMC parameters
samplerKwargs = {"nwalkers" : 20}        # emcee.EnsembleSampler parameters
mcmcKwargs = {"iterations" : int(2.0e4)} # emcee.EnsembleSampler.run_mcmc parameters

# Sample initial conditions from the prior
theta = lh.rosenbrockSample(m0)

# Evaluate forward model log likelihood + lnprior for each theta
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

# Create the the default GP which uses an ExpSquaredKernel
gp = gpUtils.defaultGP(theta, y, order=None, white_noise=0)

# Initialize object using the Wang & Li (2017) Rosenbrock function example
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=lh.rosenbrockLnprior,
                            lnlike=lh.rosenbrockLnlike,
                            priorSample=lh.rosenbrockSample,
                            bounds=bounds,
                            algorithm=algorithm)

# Run!
ap.run(m=m, nmax=nmax, estBurnin=True, nGPRestarts=1, mcmcKwargs=mcmcKwargs,
       cache=False, samplerKwargs=samplerKwargs, verbose=True, onlyLastMCMC=True)

# Check out the final posterior distribution!
import corner

# Load in chain from last iteration
samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True)

fig.savefig("finalPosterior.png", bbox_inches="tight")
```

The final distribution will look something like this:

![Final posterior probability distribution for the Wang & Li (2017) example.](paper/final_posterior.png)

Check out the [examples](https://github.com/dflemin3/approxposterior/tree/master/examples/Notebooks) directory for Jupyter Notebook examples and explanations. Check out the full [documentation](https://dflemin3.github.io/approxposterior/) for a more in-depth explanation of classes, methods, variables, and how to use the code.

Contribution
============

If you would like to contribute to this code, please feel free to fork the repository, make some edits, and open a pull request.
If you find a bug, have a suggestion, etc, please open up an issue!

Citation
========

If you use this code, please cite the following:

Fleming and VanderPlas (2018):

```bash
@ARTICLE{Fleming2018,
   author = {{Fleming}, D.~P. and {VanderPlas}, J.},
    title = "{approxposterior: Approximate Posterior Distributions in Python}",
  journal = {The Journal of Open Source Software},
     year = 2018,
    month = sep,
   volume = 3,
    pages = {781},
      doi = {10.21105/joss.00781},
   adsurl = {http://adsabs.harvard.edu/abs/2018JOSS....3..781P},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

Kandasamy et al. (2015):

```bash
@misc{Kandasamy2015,
	author = {Kirthevasan Kandasamy and Jeff Schneider and Barnabas Poczos},
	title = "{Bayesian Active Learning for Posterior Estimation}",
	note = {International Joint Conference on Artificial Intelligence},
	year = {2015},
	}
```

Wang & Li (2017):

```bash
@ARTICLE{Wang2017,
   author = {{Wang}, H. and {Li}, J.},
    title = "{Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1703.09930},
 primaryClass = "stat.CO",
 keywords = {Statistics - Computation, Statistics - Machine Learning},
     year = 2017,
    month = mar,
   adsurl = {http://adsabs.harvard.edu/abs/2017arXiv170309930W},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
