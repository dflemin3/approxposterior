# -*- coding: utf-8 -*-
"""
:py:mod:`priors.py` - Prior Objects and Functions
-------------------------------------------------

Priors are essential ingredients for Bayesian inference. See
`"Bayesian Methods for Exoplanet Science" by Parviainen (2017) <https://arxiv.org/pdf/1711.03329.pdf>`_
for an awesome exoplanet-focused introduction to Bayesian Methods, from which we
have adapted the following text: 

The role of a prior distribution is to encapsulate the current information and
assumptions about a model parameter (or a quantity that depends on the model parameters).
As new information (observations) is obtained, the prior is updated by the likelihood
to produce a posterior distribution, which can be used as a prior distribution in future
analyses.

Priors can be (roughly) classified as either informative priors or weakly
informative (uninformative) priors, depending on how strongly they constrain the parameter
space. Informative priors can be based on previous research and theory. For example,
one can use a normal distribution (:class:`GaussianPrior`) with mean and standard deviation based on
previously reported parameter mean and uncertainty estimates. Weakly informative
priors, such as a :class:`UniformPrior`, are used to express our ignorance about
a parameter, and aim to minimise the effect the prior has on the posterior,
allowing the data to "speak for itself".

Example
-------
1. Generating numerical and analytic 1D prior probability density functions:

.. code-block:: python

  from approxposterior.priors import UniformPrior, GaussianPrior

  # Number of numerical samples
  N=10000

  # Define prior distributions
  u = UniformPrior(0.0, 100.0)
  g = GaussianPrior(50.0, 10.0)

  # Plot histograms
  plt.hist(u.random_sample(N), bins = 100, density=True, alpha = 0.5, color="C0");
  plt.hist(g.random_sample(N), bins = 100, density=True, alpha = 0.5, color="C1");

  # Plot the analytic density
  xmin, xmax = plt.xlim();
  x = np.linspace(xmin, xmax, 1000);
  plt.plot(x, u.dist.pdf(x), c="C0", lw = 3.0, label="uniform")
  plt.plot(x, g.dist.pdf(x), c="C1", lw = 3.0, label="normal")

  # Tweak plot style
  plt.xlabel(r"$x$")
  plt.ylabel(r"Prior Probability Density, $\mathcal{P}(x)$")
  plt.legend(framealpha=0.0)

.. plot::
  :align: center

  import matplotlib.pyplot as plt
  import numpy as np
  from approxposterior.priors import UniformPrior, GaussianPrior

  # Number of numerical samples
  N=10000

  # Define prior distributions
  u = UniformPrior(0.0, 100.0)
  g = GaussianPrior(50.0, 10.0)

  # Plot histograms
  plt.hist(u.random_sample(N), bins = 100, density=True, alpha = 0.5, color="C0");
  plt.hist(g.random_sample(N), bins = 100, density=True, alpha = 0.5, color="C1");

  # Plot the analytic density
  xmin, xmax = plt.xlim();
  x = np.linspace(xmin, xmax, 1000);
  plt.plot(x, u.dist.pdf(x), c="C0", lw = 3.0, label="uniform")
  plt.plot(x, g.dist.pdf(x), c="C1", lw = 3.0, label="normal")

  # Tweak plot style
  plt.xlabel(r"$x$")
  plt.ylabel(r"Prior Probability Density, $\mathcal{P}(x)$")
  plt.legend(framealpha=0.0)
  plt.show()

2. Sampling from your priors:

.. code-block:: python

  from corner import corner
  from approxposterior.priors import UniformPrior, GaussianPrior, get_theta_names

  # Number of numerical samples
  N = 100000

  # Define some Gaussian Priors
  priors = [GaussianPrior(10.0,5.0, theta_name=r"$x_1$"),
            GaussianPrior(1.0,0.05, theta_name=r"$x_2$"),
            GaussianPrior(50.0,10.0, theta_name=r"$x_3$"),
            GaussianPrior(-10,1.0, theta_name=r"$x_4$")]

  # Get N samples for each prior
  samples = np.vstack([prior.random_sample(N) for prior in priors]).T

  # Create some labels
  labels = get_theta_names(priors)

  # Make corner plot
  fig = corner(samples, labels=labels);

.. plot::
  :align: center

  import matplotlib.pyplot as plt
  import numpy as np
  from corner import corner
  from approxposterior.priors import UniformPrior, GaussianPrior, get_theta_names

  # Number of numerical samples
  N = 100000

  # Define some Gaussian Priors
  priors = [GaussianPrior(10.0,5.0, theta_name=r"$x_1$"),
            GaussianPrior(1.0,0.05, theta_name=r"$x_2$"),
            GaussianPrior(50.0,10.0, theta_name=r"$x_3$"),
            GaussianPrior(-10,1.0, theta_name=r"$x_4$")]

  # Get N samples for each prior
  samples = np.vstack([prior.random_sample(N) for prior in priors]).T

  # Create some labels
  labels = get_theta_names(priors)

  # Make corner plot
  fig = corner(samples, labels=labels);
  fig.subplots_adjust(wspace = 0.05, hspace = 0.05);
  plt.show()


"""

__all__ = ["Prior", "UniformPrior", "GaussianPrior", "get_lnprior",
           "get_prior_unit_cube", "get_theta_bounds", "get_theta_names"]

# Generic packages
import numpy as np
import scipy as sp
from scipy.special import erfcinv

################################################################################
# P r i o r   C l a s s
################################################################################

class Prior(object):
    """
    Prior probability class meant for subclassing.

    Warning
    -------
    :class:`Prior` is a base class to construct specific prior distribution
    classes and instances. It cannot be used directly as a prior. See
    :class:`UniformPrior` and :class:`GaussianPrior` for functional
    subclasses.

    Parameters
    ----------
    theta_name : str
        State vector parameter name

    """
    def __init__(self, theta_name = None):
        self.theta_name = theta_name
        return

    def __call__(self, x):
        """
        Returns the log-prior probability of ``x``
        """
        return self.lnprior(x)

    def __repr__(self):
        """
        """
        return "%s(%s=%.3f, %s=%.3f)" %(self.__class__.__name__,
                                        list(self.__dict__.keys())[0],
                                        list(self.__dict__.values())[0],
                                        list(self.__dict__.keys())[1],
                                        list(self.__dict__.values())[1])

    def __str__(self):
        """
        """
        return self.__repr__()

    def lnprior(self, x):
        """
        Returns the natural log of the prior probability

        Parameters
        ----------
        x : float
            State at which to evaluate the log-prior

        Returns
        -------
        lnprior : float
            Log of the prior probability
        """
        return NotImplementedError("You must specify `lnprior` function in a subclass.")

    def random_sample(self):
        """
        Returns a sample from the prior probability distribution function

        Parameters
        ----------
        size : int or None
            Number of random samples to return; default ``None`` will return a
            float, otherwise a numpy array is returned.

        Returns
        -------
        x0 : float or numpy.array
            Randomly drawn sample from the prior
        """
        return NotImplementedError("You must specify `random_sample` function in a subclass.")


    def transform_uniform(self, r):
        """
        Tranformation from hypercube to physical parameters. The MultiNest native space is a unit hyper-cube
        in which all the parameter are uniformly distributed in [0, 1]. The user is required to transform
        the hypercube parameters to physical parameters. This transformation is described in Sec 5.1
        of arXiv:0809.3437.

        These functions are based on the prior transformations provided here:
        https://github.com/JohannesBuchner/MultiNest/blob/master/src/priors.f90

        Parameters
        ----------
        r : float
            Hypercube value

        Returns
        -------
        r2 : float
            Transformed parameter value

        """
        return NotImplementedError("`transform_uniform` must be specified by a specific subclass.")

    def get_bounds(self):
        """
        Returns a tuple of the strict boundaries

        Returns
        -------
        bounds : tuple
            Hard bounds ``(xmin, xmax)``
        """
        return NotImplementedError("You must specify `get_bounds` in a subclass.")

################################################################################
# U n i f o r m   P r i o r
################################################################################

class UniformPrior(Prior):
    """
    Uniform prior subclass. This distribution is constant between low and
    high.

    Parameters
    ----------
    low : float
        Minimum parameter value
    high : float
        Maximum parameter value

    Attributes
    ----------
    dist : scipy.stats.uniform
        A uniform continuous random variable instance
    """
    def __init__(self, low, high, **kwargs):
        self.low = low
        self.high = high

        self.dist = sp.stats.uniform(loc = self.low, scale = self.high - self.low)

        super(UniformPrior, self).__init__(**kwargs)

        return

    def lnprior(self, x):
        """
        Returns the natural log of the prior probability

        Parameters
        ----------
        x : float
            State at which to evaluate the log-prior

        Returns
        -------
        lnprior : float
            Log of the prior probability
        """

        #if x >= self.low and x <= self.high:
        #    lp = 0.0
        #else:
        #    lp = -np.inf

        return self.dist.logpdf(x) #lp

    def random_sample(self, size=None):
        """
        Returns a sample from the prior probability distribution function

        Parameters
        ----------
        size : int or None
            Number of random samples to return; default ``None`` will return a
            float, otherwise a numpy array is returned.

        Returns
        -------
        x0 : float or numpy.array
            Randomly drawn sample from the prior
        """
        return self.dist.rvs(size=size)

    def transform_uniform(self, r):
        """
        Tranformation from hypercube to physical parameters. The MultiNest native space is a unit hyper-cube
        in which all the parameter are uniformly distributed in [0, 1]. The user is required to transform
        the hypercube parameters to physical parameters. This transformation is described in Sec 5.1
        of arXiv:0809.3437.

        These functions are based on the prior transformations provided here:
        https://github.com/JohannesBuchner/MultiNest/blob/master/src/priors.f90

        Parameters
        ----------
        r : float
            Hypercube value

        Returns
        -------
        r2 : float
            Transformed parameter value
        """

        # Parse attributes
        x1 = self.low
        x2 = self.high

        # Calculate transformation
        u=x1+r*(x2-x1)

        return u

    def get_bounds(self):
        """
        Returns a tuple of the strict boundaries

        Returns
        -------
        bounds : tuple
            Hard bounds ``(xmin, xmax)``
        """
        return (self.low, self.high)

################################################################################
# G a u s s i a n   P r i o r
################################################################################

class GaussianPrior(Prior):
    """
    Gaussian prior object.

    The probability density for the Gaussian distribution is

    .. math::

      p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }} e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },

    where :math:`\mu` is the mean and :math:`\sigma` the standard
    deviation. The square of the standard deviation, :math:`\sigma^2`,
    is called the variance.

    Parameters
    ----------
    mu : float
        Mean of the normal distribution
    sigma : float
        Standard deviation of the normal distribution

    Attributes
    ----------
    dist : scipy.stats.norm
        A normal continuous random variable instance

    """
    def __init__(self, mu, sigma, **kwargs):
        self.mu = mu
        self.sigma = sigma

        self.dist = sp.stats.norm(loc = self.mu, scale = self.sigma)

        super(GaussianPrior, self).__init__(**kwargs)
        return

    def lnprior(self, x):
        """
        Returns the natural log of the prior probability

        Parameters
        ----------
        x : float
            State at which to evaluate the log-prior

        Returns
        -------
        lnprior : float
            Log of the prior probability
        """
        #p = (1.0 / (2.0 * np.pi * self.sigma**2.0)) * np.exp(- ((x - self.mu)**2.0) / (2.0 * self.sigma**2.0))
        return self.dist.logpdf(x)

    def random_sample(self, size=None):
        """
        Returns a sample from the prior probability distribution function

        Parameters
        ----------
        size : int or None
            Number of random samples to return; default ``None`` will return a
            float, otherwise a numpy array is returned.

        Returns
        -------
        x0 : float or numpy.array
            Randomly drawn sample from the prior
        """
        return self.dist.rvs(size=size)

    def transform_uniform(self, r):
        """
        Tranformation from hypercube to physical parameters. The MultiNest native space is a unit hyper-cube
        in which all the parameter are uniformly distributed in [0, 1]. The user is required to transform
        the hypercube parameters to physical parameters. This transformation is described in Sec 5.1
        of arXiv:0809.3437.

        These functions are based on the prior transformations provided here:
        https://github.com/JohannesBuchner/MultiNest/blob/master/src/priors.f90

        Parameters
        ----------
        r : float
            Hypercube value

        Returns
        -------
        r2 : float
            Transformed parameter value
        """

        # Calculate transformation
        u = self.mu + self.sigma * np.sqrt(2.0) * erfcinv(2.0*(1.0 - r))

        return u

    def get_bounds(self, Nstd = 5.0):
        """
        Returns a tuple of the strict boundaries

        Parameters
        ----------
        Nstd : float, optional
            Number of standard deviations away from the mean to define hard bounds

        Returns
        -------
        bounds : tuple
            Hard bounds ``(xmin, xmax)``
        """
        return (self.dist.mean() - Nstd*self.dist.std(), self.dist.mean() + Nstd*self.dist.std())

################################################################################
# P r i o r  U t i l i t i e s
################################################################################

def get_lnprior(theta, priors):
    """
    Returns the summed log-prior probability of ``theta`` given ``priors``.

    Parameters
    ----------
    theta : list
        State vector
    priors : list of Prior
        :class:`Prior` vector

    Returns
    -------
    lp : int
        Log-prior probability
    """

    assert len(theta) == len(priors)

    lp = 0.0

    # Loop over all parameters
    for i, prior in enumerate(priors):

        # Sum lnprobs
        lp += prior.lnprior(theta[i])

    return lp

def get_prior_unit_cube(cube, priors):
    """
    Returns the transformed unit cube for MultiNest.

    Parameters
    ----------
    cube : list or numpy.array
        Unit cube [0,1]
    priors : list of instantiated Prior objects
        :class:`Prior` vector

    Returns
    -------
    cube : list or numpy.array
        Physical parameters
    """

    # Loop over all parameters
    for i, prior in enumerate(priors):

        # Transform from uniform to physical
        cube[i] = prior.transform_uniform(cube[i])

    return cube

def get_theta_bounds(priors):
    """
    Returns the state vector parameters bounds.

    Parameters
    ----------
    priors : list of instantiated Prior objects
        :class:`Prior` vector

    Returns
    -------
    bounds : list
        List of (min, max) tuples
    """
    return [prior.get_bounds() for prior in priors]

def get_theta_names(priors):
    """
    Returns a list of state vector names.

    Parameters
    ----------
    priors : list of instantiated Prior objects
        :class:`Prior` vector

    Returns
    -------
    theta_names : list
        List of parameter names
    """
    return [prior.theta_name for prior in priors]
