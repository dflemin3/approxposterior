"""

Bayesian Posterior estimation routines written in pure python leveraging
Dan Forman-Mackey's george Gaussian Process implementation and emcee.

August 2017

@author: David P. Fleming [University of Washington, Seattle]
@email: dflemin3 (at) uw (dot) edu

A really shitty implementation of Kandasamy et al. (2015)'s BAPE model.

TODO
    - utility module to make sure values are sane (not inf, nan, etc)

"""

# Tell module what it's allowed to import
__all__ = ["ApproxPosterior"]

from __future__ import (print_function, division, absolute_import, unicode_literals)
import numpy as np
import george
from george import kernels
import emcee
import corner
from scipy.optimize import minimize, basinhopping
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def logsubexp(x1, x2):
    """
    More numerically stable way to take the log of exp(x1) - exp(x2)
    via:

    logsubexp(x1, x2) -> log(exp(x1) - exp(x2))

    Parameters
    ----------
    x1 : float
    x2 : float

    Returns
    -------
    logsubexp(x1, x2)
    """

    if x1 <= x2:
        return -np.inf
    else:
        return x1 + np.log(1.0 - np.exp(x2 - x1))
# end function


def rosenbrock_log_likelihood(x):
    """
    2D Rosenbrock function as a log likelihood following Wang & Li (2017)

    Parameters
    ----------
    x : array

    Returns
    -------
    l : float
        likelihood
    """

    x = np.array(x)
    if x.ndim > 1:
        x1 = x[:,0]
        x2 = x[:,1]
    else:
        x1 = x[0]
        x2 = x[1]

    return -0.01*(x1 - 1.0)**2 - (x1*x1 - x2)**2
# end function

def log_rb_prior(x1, x2):
    """
    Uniform log prior for the 2D Rosenbrock likelihood following Wang & Li (2017)
    where the prior pi(x) is a uniform distribution over [-5, 5] x [-5, 5]

    Parameters
    ----------
    x : array

    Returns
    -------
    l : float
        log prior
    """
    if (x1 > 5) or (x1 < -5) or (x2 > 5) or (x2 < -5):
        return -np.inf

    # All parameters in range equally likely
    return 0.0
log_rb_prior = np.vectorize(log_rb_prior)
# end function


def log_rosenbrock_prior(x):
    """
    Uniform log prior for the 2D Rosenbrock likelihood following Wang & Li (2017)
    where the prior pi(x) is a uniform distribution over [-5, 5] x [-5, 5]

    Parameters
    ----------
    x : array

    Returns
    -------
    l : float
        log prior
    """

    x = np.array(x)
    if x.ndim > 1:
        x1 = x[:,0]
        x2 = x[:,1]
    else:
        x1 = x[0]
        x2 = x[1]

    return log_rb_prior(x1, x2)
# end function


def rosenbrock_prior(x):
    """
    Uniform prior for the 2D Rosenbrock likelihood following Wang & Li (2017)
    where the prior pi(x) is a uniform distribution over [-5, 5] x [-5, 5]

    Parameters
    ----------
    x : array

    Returns
    -------
    l : float
        log prior
    """

    return np.exp(log_rosenbrock_prior(x))
# end function


def rosenbrock_sample(n):
    """
    Sample N points from the prior pi(x) is a uniform distribution over
    [-5, 5] x [-5, 5]

    Parameters
    ----------
    n : int
        Number of samples

    Returns
    -------
    sample : floats
        n x 2 array of floats samples from the prior
    """

    return np.random.uniform(low=-5, high=5, size=(n,2)).squeeze()
# end function


def AGP_utility(theta, y, gp):
    """
    AGP (Adaptive Gaussian Process) utility function, the entropy of the posterior
    distribution. This is what you maximize to find the next x under the AGP
    formalism. Note here we use the negative of the utility function so
    minimizing this is the same as maximizing the actual utility function.

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object

    Returns
    -------
    u : float
        utility of theta under the gp
    """

    # Only works if the GP object has been computed, otherwise you messed up
    if gp.computed:
        mu, var = gp.predict(y, theta.reshape(1,-1), return_var=True)
    else:
        raise RuntimeError("ERROR: Need to compute GP before using it!")

    return -(mu + 1.0/np.log(2.0*np.pi*np.e*var))
    #return -(mu + np.log(2.0*np.pi*np.e*var))
# end function


def BAPE_utility(theta, y, gp):
    """
    BAPE (Bayesian Active Posterior Estimation) utility function.  This is what
    you maximize to find the next theta under the BAPE formalism.  Note here we use
    the negative of the utility function so minimizing this is the same as
    maximizing the actual utility function.  Also, we log the BAPE utility
    function as the log is monotonic so the minima are equivalent.

    Parameters
    ----------
    theta : array
        parameters to evaluate
    y : array
        y values to condition the gp prediction on.
    gp : george GP object

    Returns
    -------
    u : float
        utility of theta under the gp
    """

    # Only works if the GP object has been computed, otherwise you messed up
    if gp.computed:
        mu, var = gp.predict(y, theta.reshape(1,-1), return_var=True)
    else:
        raise RuntimeError("ERROR: Need to compute GP before using it!")

    return -((2.0*mu + var) + logsubexp(var, 0.0))
# end function


def minimize_objective(fn, y, gp, sample_fn=None, prior_fn=None,
                       sim_annealing=False, **kw):
    """
    Find point that minimizes fn for a gaussian process gp conditioned on y,
    the data.

    Parameters
    ----------
    fn : function
        function to minimize that expects x, y, gp as arguments aka fn looks like
        fn_name(x, y, gp).  See *_utility functions above for examples.
    y : array
        y values to condition the gp prediction on.
    gp : george GP object
    sample_fn : function (optional)
        Function to sample initial conditions from.  Defaults to None, so we'd
        use rosenbrock_sample
    prior_fn : function (optional)
        Function to apply prior to.  If sample is rejected by prior, reject sample
        and try again.
    sim_annealing : bool (optional)
        Whether to use the simulated annealing (basinhopping) algorithm.  Defaults
        to False.
    kw : dict (optional)
        Any additional keyword arguments scipy.optimize.minimize could use, e.g.,
        method.

    Returns
    -------
    theta : (1 x n_dims)
        point that minimizes fn
    """

    # Assign sampling, prior function if it's not provided
    if sample_fn is None:
        sample_fn = rosenbrock_sample

    # Assign prior function if it's not provided
    if prior_fn is None:
        prior_fn = log_rosenbrock_prior

    is_finite = False
    while not is_finite:
        # Solve for theta that maximize fn and is allowed by prior

        # Choose theta0 by uniformly sampling over parameter space
        theta0 = sample_fn(1).reshape(1,-1)

        args=(y, gp)

        bounds = ((-5,5), (-5,5))
        #bounds = None

        # Mimimze fn, see if prior allows solution
        try:
            if sim_annealing:
                minimizer_kwargs = {"method":"L-BFGS-B", "args" : args,
                                    "bounds" : bounds, "options" : {"ftol" : 1.0e-3}}

                def mybounds(**kwargs):
                    x = kwargs["x_new"]
                    res = bool(np.all(np.fabs(x) < 5))
                    return res

                tmp = basinhopping(fn, theta0, accept_test=mybounds, niter=500,
                             stepsize=0.01, minimizer_kwargs=minimizer_kwargs,
                             interval=10)["x"]
            else:
                tmp = minimize(fn, theta0, args=args, bounds=bounds,
                               method="l-bfgs-b", options={"ftol" : 1.0e-3},
                               **kw)["x"]

        # ValueError.  Try again.
        except ValueError:
            tmp = np.array([np.inf for ii in range(theta0.shape[-1])]).reshape(theta0.shape)
        if np.isfinite(prior_fn(tmp).all()) and not np.isinf(tmp).any() and not np.isnan(tmp).any() and np.isfinite(tmp.sum()):
            theta = tmp
            is_finite = True
    # end while

    return np.array(theta).reshape(1,-1)
# end function


# Define the objective function (negative log-likelihood in this case).
def nll(p, gp, y):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25
# end function


# And the gradient of the objective function.
def grad_nll(p, gp, y):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)
# end function


def optimize_gp(gp, y):
    """
    DOCS. So messy, must clean up if this works

    Optimize hyperparameters of pre-computed gp
    """

    # Run the optimization routine.
    p0 = gp.get_parameter_vector()
    results = minimize(nll, p0, jac=grad_nll, args=(gp, y), method="bfgs")

    # Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)
    gp.recompute()
# end function


def plot_gp(gp, theta, y, xmin=-5, xmax=5, ymin=-5, ymax=5, n=100,
            return_var=False, save_plot=None, log=False, **kw):
    """
    DOCS
    """

    xx = np.linspace(xmin, xmax, n)
    yy = np.linspace(ymin, ymax, n)

    zz = np.zeros((len(xx),len(yy)))
    for ii in range(len(xx)):
        for jj in range(len(yy)):
            mu, var = gp.predict(y, np.array([xx[ii],yy[jj]]).reshape(1,-1), return_var=return_var)
            if return_var:
                zz[ii,jj] = var
            else:
                zz[ii,jj] = mu

    if log:
        if not return_var:
            zz = np.fabs(zz)
            #norm = LogNorm(vmin=1.0e-4, vmax=1.0e2)
        if return_var:

            zz[zz < 1.0e-6] = 1.0e-1
            #norm = LogNorm(vmin=1.0e-1, vmax=1.0e5)

        norm = LogNorm(vmin=zz.min(), vmax=zz.max())


    # Plot what the GP thinks the function looks like
    fig, ax = plt.subplots(**kw)
    im = ax.pcolormesh(xx, yy, zz.T, norm=norm)
    cb = fig.colorbar(im)

    if return_var:
        cb.set_label("Variance", labelpad=20, rotation=270)
    else:
        cb.set_label("|Mean|", labelpad=20, rotation=270)


    # Scatter plot where the points are
    ax.scatter(theta[:,0], theta[:,1], color="red")

    # Format
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if save_plot is not None:
        fig.savefig(save_plot, bbox_inches="tight")

    return fig, ax
# end function


class ApproxPosterior(object):
    """
    Class to approximate the posterior distributions using either the
    Bayesian Active Posterior Estimation (BAPE) by Kandasamy et al. 2015 or the
    AGP (Adaptive Gaussian Process) by XXX et al.
    """

    def __init__(self, gp, prior, loglike, algorithm="BAPE"):
        """
        Initializer.

        Parameters
        ----------
        gp : george.GP
            Gaussian process object
        prior : function
            Defines the log prior over the input features.
        loglike : function
            Defines the log likelihood function.  In this function, it is assumed
            that the forward model is evaluated on the input theta and the output
            is used to evaluate the log likelihood.
        algorithm : str (optional)
            Which utility function to use.  Defaults to BAPE.

        Returns
        -------
        None
        """

        self.gp = gp
        self.prior = prior
        self._loglike = loglike
        self.algorithm = algorithm

        # Assign utility function
        if self.algorithm.lower() == "bape":
            self.utility = BAPE_utility
        elif self.algorithm.lower() == "agp":
            self.utility = AGP_utility
        else:
            raise IOError("Invalid algorithm. Valid options: BAPE, AGP.")

        # Initial approximate posteriors are the prior
        self.posterior = prior
        self.__prev_posterior = prior

    # end function


    def _sample(self, theta):
        """
        Draw a sample from the approximate posterior
        DOCS
        """
        theta_test = np.array(theta).reshape(1,-1)

        # Sometimes the input values can be crazy
        if np.isinf(theta_test).any() or np.isnan(theta_test).any() or not np.isfinite(theta_test.sum()):
            return -np.inf

        res = self.gp.sample_conditional(self.__y, theta_test) + self.posterior(theta_test)

        # Catch NaNs because they can happen for I don't know why reasons
        if np.isnan(res):
            return -np.inf
        else:
            return res
    # end function


    def run(self, theta, y, m=10, M=10000, nmax=2, Dmax=0.1, kmax=5, sampler=None,
            sim_annealing=False, **kw):
        """
        Core algorithm.

        Parameters
        ----------
        theta : array
            Input features (n_samples x n_features)
        y : array
            Input result of forward model (n_samples,)
        m : int (optional)
            Number of new input features to find each iteration.  Defaults to 10.
        M : int (optional)
            Number of MCMC steps to sample GP to estimate the approximate posterior.
            Defaults to 10^4.
        nmax : int (optional)
            Maximum number of iterations.  Defaults to 2 for testing.
        Dmax : float (optional)
            Maximum change in KL divergence for convergence checking.  Defaults to 0.1.
        kmax : int (optional)
            Maximum number of iterators such that if the change in KL divergence is
            less than Dmax for kmax iterators, the algorithm is considered
            converged and terminates.  Defaults to 5.
        sample : emcee.EnsembleSampler (optional)
            emcee sampler object.  Defaults to None and is initialized internally.
        sim_annealing : bool (optional)
            Whether or not to minimize utility function using simulated annealing.
            Defaults to False.

        Returns
        -------
        None
        """

        # Store theta, y
        self.__theta = theta
        self.__y = y

        # Main loop
        for n in range(nmax):

            # 1) Find m new points by maximizing utility function
            for ii in range(m):
                theta_t = minimize_objective(self.utility, self.__y, self.gp,
                                             sample_fn=None,
                                             sim_annealing=sim_annealing,
                                             **kw)

                # 2) Query oracle at new points, theta_t
                y_t = self._loglike(theta_t) - self.posterior(theta_t)

                # Join theta, y arrays
                self.__theta = np.concatenate([self.__theta, theta_t])
                self.__y = np.concatenate([self.__y, y_t])

                # 3) Refit GP
                # Guess the bandwidth following Kandasamy et al. (2015)'s suggestion
                bandwidth = 5 * np.power(len(self.__y),(-1.0/self.__theta.shape[-1]))

                # Create the GP conditioned on {theta_n, log(L_n * p_n)}
                kernel = np.var(self.__y) * kernels.ExpSquaredKernel(bandwidth, ndim=self.__theta.shape[-1])
                self.gp = george.GP(kernel)
                self.gp.compute(self.__theta)

                # Optimize gp hyperparameters
                optimize_gp(self.gp, self.__y)

            # Done adding new design points
            fig, _ = plot_gp(self.gp, self.__theta, self.__y, return_var=False,
                    save_plot="gp_mu_iter_%d.png" % n, log=True)
            plt.close(fig)

            # Done adding new design points
            fig, _ = plot_gp(self.gp, self.__theta, self.__y, return_var=True,
                    save_plot="gp_var_iter_%d.png" % n, log=True)
            plt.close(fig)

            # GP updated: run sampler to obtain new posterior conditioned on (theta_n, log(L_t)*p_n)

            # Use emcee to obtain approximate posterior
            ndim = self.__theta.shape[-1]
            nwalk = 10 * ndim
            nsteps = M

            # Initial guess (random over interval)
            p0 = [np.random.uniform(low=-5, high=5, size=ndim) for j in range(nwalk)]
            #p0 = [np.random.rand(ndim) for j in range(nwalk)]
            params = ["x%d" % jj for jj in range(ndim)]
            sampler = emcee.EnsembleSampler(nwalk, ndim, self._sample)
            for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
                print("%d/%d" % (i+1, nsteps))

            print("emcee finished!")

            #fig = corner.corner(sampler.flatchain, quantiles=[0.16, 0.5, 0.84],
            #                    plot_contours=False, bins="auto");
            fig, ax = plt.subplots(figsize=(9,8))
            corner.hist2d(sampler.flatchain[:,0], sampler.flatchain[:,1], ax=ax,
                                plot_contours=False, no_fill_contours=True,
                                plot_density=True)
            ax.scatter(self.__theta[:,0], self.__theta[:,1])
            ax.set_xlim(-5,5)
            ax.set_ylim(-5,5)
            fig.savefig("posterior_%d.png" % n)
            #plt.show()

            # Make new posterior function via a Gaussian Mixure model approximation
            # to the approximate posterior. Seems legit
            # Fit some GMMs!
            # sklean hates infs, Nans, big numbers
            mask = (~np.isnan(sampler.flatchain).any(axis=1)) & (~np.isinf(sampler.flatchain).any(axis=1))
            bic = []
            lowest_bic = 1.0e10
            best_gmm = None
            gmm = GaussianMixture()
            for n in range(5,10):
                gmm.set_params(**{"n_components" : n, "covariance_type" : "full"})
                gmm.fit(sampler.flatchain[mask])
                bic.append(gmm.bic(sampler.flatchain[mask]))

                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

            # Refit GMM with the lowest bic
            GMM = best_gmm
            GMM.fit(sampler.flatchain[mask])
            #self.posterior = GMM.score_samples

# Test!
if __name__ == "__main__":

    # Define algorithm parameters
    m0 = 20 # Initialize size of training set
    m = 10  # Number of new points to find each iteration
    nmax = 10 # Maximum number of iterations
    M = int(1.0e2) # Number of MCMC steps to estimate approximate posterior
    Dmax = 0.1
    kmax = 5
    kw = {}

    # Choose m0 initial design points to initialize dataset
    theta = rosenbrock_sample(m0)
    y = rosenbrock_log_likelihood(theta) + log_rosenbrock_prior(theta)

    # 0) Initial GP fit
    # Guess the bandwidth following Kandasamy et al. (2015)'s suggestion
    bandwidth = 5 * np.power(len(y),(-1.0/theta.shape[-1]))

    # Create the GP conditioned on {theta_n, log(L_n / p_n)}
    kernel = np.var(y) * kernels.ExpSquaredKernel(bandwidth, ndim=theta.shape[-1])
    gp = george.GP(kernel)
    gp.compute(theta)

    # Optimize gp hyperparameters
    optimize_gp(gp, y)

    # Init object
    bp = ApproxPosterior(gp, prior=log_rosenbrock_prior,
                         loglike=rosenbrock_log_likelihood,
                         algorithm="agp")

    # Run this bastard
    bp.run(theta, y, m=m, M=M, nmax=nmax, Dmax=Dmax, kmax=kmax,
           sampler=None, sim_annealing=False, **kw)
