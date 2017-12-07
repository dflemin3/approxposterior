"""

Bayesian Posterior estimation routines written in pure python leveraging
Dan Forman-Mackey's george Gaussian Process implementation and emcee.

August 2017

@author: David P. Fleming [University of Washington, Seattle]
@email: dflemin3 (at) uw (dot) edu

A meh implementation of Kandasamy et al. (2015)'s BAPE model.

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# Tell module what it's allowed to import
__all__ = ["ApproxPosterior"]

from . import utility as ut
from . import likelihood as lh
import numpy as np
import george
from george import kernels
import emcee
import corner
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_gp(gp, theta, y, xmin=-5, xmax=5, ymin=-5, ymax=5, n=100,
            return_type="mean", save_plot=None, log=False, **kw):
    """
    DOCS
    """

    xx = np.linspace(xmin, xmax, n)
    yy = np.linspace(ymin, ymax, n)

    zz = np.zeros((len(xx),len(yy)))
    for ii in range(len(xx)):
        for jj in range(len(yy)):
            mu, var = gp.predict(y, np.array([xx[ii],yy[jj]]).reshape(1,-1), return_var=True)
            if return_type.lower() == "var":
                zz[ii,jj] = var
            elif return_type.lower() == "mean":
                zz[ii,jj] = mu
            elif return_type.lower() == "utility":
                zz[ii,jj] = np.fabs(-(2.0*mu + var) - ut.logsubexp(var, 0.0))
            else:
                raise IOError("Invalid return_type : %s" % return_type)

    norm = None
    if log:
        if return_type.lower() == "mean" or return_type.lower() == "utility":
            zz = np.fabs(zz)
            zz[zz <= 1.0e-5] = 1.0e-5

        if return_type.lower() == "var":
            zz[zz <= 1.0e-8] = 1.0e-8

        norm = LogNorm(vmin=zz.min(), vmax=zz.max())

    # Plot what the GP thinks the function looks like
    fig, ax = plt.subplots(**kw)
    im = ax.pcolormesh(xx, yy, zz.T, norm=norm)
    cb = fig.colorbar(im)

    if return_type.lower() == "var":
        cb.set_label("GP Posterior Variance", labelpad=20, rotation=270)
    elif return_type.lower() == "mean":
        cb.set_label("|Mean GP Posterior Density (smaller better)|", labelpad=20, rotation=270)
    elif return_type.lower() == "utility":
        cb.set_label("|Utility Function (smaller better)|", labelpad=20, rotation=270)

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

    def __init__(self, lnprior, lnlike, lnprob, prior_sample, algorithm="BAPE"):
        """
        Initializer.

        Parameters
        ----------
        lnprior : function
            Defines the log prior over the input features.
        lnlike : function
            Defines the log likelihood function.  In this function, it is assumed
            that the forward model is evaluated on the input theta and the output
            is used to evaluate the log likelihood.
        lnprob : function
            Defines the log probability function
        prior_sample : function
            Method to randomly sample points over region allowed by prior
        algorithm : str (optional)
            Which utility function to use.  Defaults to BAPE.

        Returns
        -------
        None
        """

        self._lnprior = lnprior
        self._lnlike = lnlike
        self._lnprob = lnprob
        self.prior_sample = prior_sample
        self.algorithm = algorithm

        # Store GPs, samplers XXX

        # Assign utility function
        if self.algorithm.lower() == "bape":
            self.utility = ut.BAPE_utility
        elif self.algorithm.lower() == "agp":
            self.utility = ut.AGP_utility
        else:
            raise IOError("Invalid algorithm. Valid options: BAPE, AGP.")

        # Initial approximate posteriors are the prior
        self.posterior = self._lnprior
        self.__prev_posterior = self._lnprior

    # end function


    def _sample(self, theta):
        """
        Draw a sample from the approximate posterior conditional distibution
        DOCS
        """
        theta_test = np.array(theta).reshape(1,-1)

        # Sometimes the input values can be crazy and the GP will blow up
        if np.isinf(theta_test).any() or np.isnan(theta_test).any():
            return -np.inf

        #mu = self.gp.sample_conditional(self.__y, theta_test) + self.posterior(theta_test)
        # Mean of predictive distribution conditioned on y (GP posterior estimate)
        mu = self.gp.predict(self.__y, theta_test, return_cov=False, return_var=False) + self.posterior(theta_test)

        # Always add flat prior to keep it in bounds
        mu += self._lnprior(theta_test)


        # Catch NaNs/Infs because they can (rarely) happen
        if not np.isfinite(mu):
            return -np.inf
        else:
            return mu
    # end function


    def run(self, theta=None, y=None, m0=20, m=10, M=10000, nmax=2, Dmax=0.1,
            kmax=5, sampler=None, sim_annealing=False, **kw):
        """
        Core algorithm.

        Parameters
        ----------
        theta : array (optional)
            Input features (n_samples x n_features).  Defaults to None.
        y : array (optional)
            Input result of forward model (n_samples,). Defaults to None.
        m0 : int (optional)
            Initial number of design points.  Defaults to 20.
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

        # Choose m0 initial design points to initialize dataset if none are
        # given
        if theta is None:
            theta = self.prior_sample(m0)

        if y is None:
            y = self._lnprob(theta)

        # 0) Initial GP fit
        # Guess the bandwidth following Kandasamy et al. (2015)'s suggestion
        bandwidth = 5 * np.power(len(y),(-1.0/theta.shape[-1]))

        # Create the GP conditioned on {theta_n, log(L_n / p_n)}
        kernel = np.var(y) * kernels.ExpSquaredKernel(bandwidth, ndim=theta.shape[-1])
        gp = george.GP(kernel)
        gp.compute(theta)

        # Optimize gp hyperparameters
        ut.optimize_gp(gp, y)

        self.gp = gp

        # Store theta, y
        self.__theta = theta
        self.__y = y

        # Main loop
        for n in range(nmax):

            # 1) Find m new points by maximizing utility function
            for ii in range(m):
                theta_t = ut.minimize_objective(self.utility, self.__y, self.gp,
                                                sample_fn=self.prior_sample,
                                                prior_fn=self._lnprior,
                                                sim_annealing=sim_annealing,
                                                **kw)

                # 2) Query oracle at new points, theta_t
                y_t = self._lnlike(theta_t) + self.posterior(theta_t)

                # Join theta, y arrays
                self.__theta = np.concatenate([self.__theta, theta_t])
                self.__y = np.concatenate([self.__y, y_t])

                # 3) Refit GP
                # Guess the bandwidth following Kandasamy et al. (2015)'s suggestion
                bandwidth = 5 * np.power(len(self.__y),(-1.0/self.__theta.shape[-1]))

                # XXX use cross-validation to select kernel parameters?

                # Create the GP conditioned on {theta_n, log(L_n * p_n)}
                kernel = kernels.ExpSquaredKernel(bandwidth, ndim=self.__theta.shape[-1])
                self.gp = george.GP(kernel)
                self.gp.compute(self.__theta)

                # Optimize gp hyperparameters
                ut.optimize_gp(self.gp, self.__y)

            # Done adding new design points
            fig, _ = plot_gp(self.gp, self.__theta, self.__y, return_type="mean",
                    save_plot="gp_mu_iter_%d.png" % n, log=True)
            plt.close(fig)

            # Done adding new design points
            fig, _ = plot_gp(self.gp, self.__theta, self.__y, return_type="var",
                    save_plot="gp_var_iter_%d.png" % n, log=True)
            plt.close(fig)

            fig, _ = plot_gp(self.gp, self.__theta, self.__y, return_type="utility",
                    save_plot="gp_util_iter_%d.png" % n, log=False)
            plt.close(fig)

            # GP updated: run sampler to obtain new posterior conditioned on (theta_n, log(L_t)*p_n)
            # Use emcee to obtain approximate posterior
            ndim = self.__theta.shape[-1]
            nwalk = 10 * ndim
            nsteps = M

            # Initial guess (random over interval)
            p0 = [self.prior_sample(1) for j in range(nwalk)]
            #p0 = [np.random.rand(ndim) for j in range(nwalk)]
            params = ["x%d" % jj for jj in range(ndim)]
            sampler = emcee.EnsembleSampler(nwalk, ndim, self._sample)
            for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
                print("%d/%d" % (i+1, nsteps))

            print("emcee finished!")
            fig = corner.corner(sampler.flatchain, quantiles=[0.16, 0.5, 0.84],
                                plot_contours=False);
            #fig, ax = plt.subplots(figsize=(9,8))
            #corner.hist2d(sampler.flatchain[:,0], sampler.flatchain[:,1], ax=ax,
            #                    plot_contours=False, no_fill_contours=True,
            #                    plot_density=True)
            #ax.scatter(self.__theta[:,0], self.__theta[:,1])
            #ax.set_xlim(-5,5)
            #ax.set_ylim(-5,5)
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
            for n in range(2,10):
                gmm.set_params(**{"n_components" : n, "covariance_type" : "diag"})
                gmm.fit(sampler.flatchain[mask])
                bic.append(gmm.bic(sampler.flatchain[mask]))

                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

            # Refit GMM with the lowest bic
            GMM = best_gmm
            print(best_gmm)
            GMM.fit(sampler.flatchain[mask])

            # Update posterior estimate
            self.__prev_posterior = self.posterior
            self.posterior = GMM.score # NEED TO RETURN - of this
