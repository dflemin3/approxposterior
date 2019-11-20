**approxposterior**
===================

***A Python package for approximate Bayesian inference with computationally-expensive models***

<p>
<a href="https://github.com/dflemin3/approxposterior">
<img src="https://img.shields.io/badge/GitHub-dflemin3%2Fapproxposterior-blue.svg?style=flat"></a>
<a href="https://github.com/dflemin3/approxposterior/blob/master/LICENSE">
<img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat"></a>
<a href="https://travis-ci.org/dflemin3/approxposterior">
<img src="http://img.shields.io/travis/dflemin3/approxposterior/master.svg?style=flat"></a>
<a href="https://doi.org/10.21105/joss.00781">
<img src="http://joss.theoj.org/papers/10.21105/joss.00781/status.svg"></a>
<a href="https://pypi.python.org/pypi/approxposterior/">
<img src="https://img.shields.io/pypi/pyversions/ansicolortags.svg"></a>
</p>

Overview
========

*approxposterior* is a Python package for efficient approximate Bayesian
inference and optimization of computationally-expensive models. :py:obj:`approxposterior`
trains a Gaussian process (GP) surrogate for the computationally-expensive model
and employs an active learning approach to iteratively improve the GPs predictive
performance while minimizing the number of calls to the expensive model required
to generate the GP's training set.

*approxposterior* implements both the [Bayesian Active Learning for Posterior Estimation (BAPE, Kandasamy et al. (2015))](https://www.cs.cmu.edu/~kkandasa/pubs/kandasamyIJCAI15activePostEst.pdf) and [Adaptive Gaussian process approximation for Bayesian inference with expensive likelihood functions (AGP, Wang & Li (2017))](https://arxiv.org/abs/1703.09930) algorithms for estimating posterior probability distributions for use with inference problems with computationally-expensive models. In such situations,
the goal is to infer posterior probability distributions for model parameters, given some data, with the additional constraint of minimizing the number of forward model evaluations given the model's assumed large computational cost.  *approxposterior* trains a Gaussian Process (GP) surrogate model for the likelihood evaluation by modeling the covariances in logprobability (logprior + loglikelihood) space. *approxposterior* then uses this GP within an MCMC sampler for each likelihood evaluation to perform the inference. *approxposterior* iteratively improves the GP's predictive performance by leveraging the inherent uncertainty in the GP's predictions to identify high-likelihood regions in parameter space where the GP is uncertain.  *approxposterior* then evaluates the forward model at these points to expand the training set in relevant regions of parameter space, re-training the GP to maximize its predictive ability while minimizing the size of the training set.  Check out [the BAPE paper](https://www.cs.cmu.edu/~kkandasa/pubs/kandasamyIJCAI15activePostEst.pdf) by Kandasamy et al. (2015) and [the AGP paper](https://arxiv.org/abs/1703.09930) by Wang & Li (2017) for in-depth descriptions of the respective algorithms.

In practice, we find that *approxposterior* can estimate posterior probability distributions that are accurate
approximations to the true, underlying distributions with only of order 100s-1000s model evaluations to train the GP, compared to 1,000,000, often more, required by MCMC methods, depending on the inference problem. The estimated marginal posterior distributions have medians that are all typically within a few percent of the true values, with similar uncertainties to the true distributions.  We have validated *approxposterior* for 2-5 dimensional problems, while Kandasamy et al. (2015) found in an 9-dimensional case that the BAPE algorithm significantly outperformed MCMC methods in terms of both accuracy and speed. See their paper for details and check out the examples for more information and example use cases.

Documentation
=============

Check out the documentation at [https://dflemin3.github.io/approxposterior/](https://dflemin3.github.io/approxposterior/) for a more in-depth explanation about the code, detailed API notes, numerous examples.

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
================

Below is a simple application of *approxposterior* based on the Wang & Li (2017) example.

```python
from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
import numpy as np

# Define algorithm parameters
m0 = 50                           # Initial size of training set
m = 20                            # Number of new points to find each iteration
nmax = 2                          # Maximum number of iterations
bounds = ((-5,5), (-5,5))         # Prior bounds
algorithm = "bape"                # Use the Kandasamy et al. (2015) formalism
seed = 57                         # RNG seed

np.random.seed(seed)

# emcee MCMC parameters
samplerKwargs = {"nwalkers" : 20}        # emcee.EnsembleSampler parameters
mcmcKwargs = {"iterations" : int(2.0e4)} # emcee.EnsembleSampler.run_mcmc parameters

# Sample initial conditions from prior
theta = lh.rosenbrockSample(m0)

# Evaluate forward model log likelihood + lnprior for each theta
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

# Create the the default GP which uses an ExpSquaredKernel
gp = gpUtils.defaultGP(theta, y)

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
ap.run(m=m, nmax=nmax, estBurnin=True, nGPRestarts=5, mcmcKwargs=mcmcKwargs,
       cache=False, samplerKwargs=samplerKwargs, verbose=True, onlyLastMCMC=True)

# Check out the final posterior distribution!
import corner

# Load in chain from last iteration
samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True)

# Plot where forward model was evaluated
fig.axes[2].scatter(ap.theta[m0:,0], ap.theta[m0:,1], s=10, color="red", zorder=20)

# Save figure
fig.savefig("finalPosterior.png", bbox_inches="tight")
```

The final distribution will look something like this:

![final_posterior](doc/_figures/final_posterior.png)

The red points were selected by *approxposterior* by maximizing the BAPE utility function.
At each red point, *approxposterior* ran the forward model to evaluate the true likelihood
and added this input-likelihood pair to the GP's training set, re-training the GP each time
to improve its predictive ability. Note how the points are selected in regions of
high posterior density, precisely where we would want to maximize the GP's predictive ability! By using the
BAPE point selection scheme, *approxposterior* does not waste computational resources by
evaluating the forward model in low likelihood regions.

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
