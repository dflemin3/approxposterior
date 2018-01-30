"""

Bayesian Posterior estimation routines written in pure python leveraging
Dan Forman-Mackey's george Gaussian Process implementation and emcee.

@author: David P. Fleming [University of Washington, Seattle], 2017
@email: dflemin3 (at) uw (dot) edu

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# Tell module what it's allowed to import
__all__ = ["ApproxPosterior"]

from . import utility as ut
from . import likelihood as lh
from . import gp_utils
from . import mcmc_utils
from . import plot_utils as pu
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


class ApproxPosterior(object):
    """
    Class to approximate the posterior distributions using either the
    Bayesian Active Posterior Estimation (BAPE) by Kandasamy et al. (2015) or
    the AGP (Adaptive Gaussian Process) by Wang & Li (2017).
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
            Which utility function to use.  Defaults to BAPE.  Options are BAPE
            or AGP.  Case doesn't matter.

        Returns
        -------
        None
        """

        self._lnprior = lnprior
        self._lnlike = lnlike
        self._lnprob = lnprob
        self.prior_sample = prior_sample
        self.algorithm = algorithm

        # Assign utility function
        if self.algorithm.lower() == "bape":
            self.utility = ut.BAPE_utility
        elif self.algorithm.lower() == "agp":
            self.utility = ut.AGP_utility
        else:
            err_msg = "ERROR: Invalid algorithm. Valid options: BAPE, AGP."
            raise IOError(err_msg)

        # Initial approximate posteriors are the prior
        self.posterior = self._lnprior
        self.prev_posterior = self._lnprior

        # Holders to save GMM fits to posteriors, raw samplers, KL divergences
        self.Dkl = list()
        self.GMMs = list()
        self.samplers = list()
        self.iburns = list()
    # end function


    def _sample(self, theta):
        """
        Compute the approximate posterior conditional distibution at a given
        point, theta.

        Parameters
        ----------
        theta : array-like
            Test point to evaluate GP posterior conditional distribution

        Returns
        -------
        mu : float
            Mean of predicted GP conditional posterior estimate at theta
        """
        theta_test = np.array(theta).reshape(1,-1)

        # Sometimes the input values can be crazy and the GP will blow up
        if np.isinf(theta_test).any() or np.isnan(theta_test).any():
            return -np.inf

        # Mean of predictive distribution conditioned on y (GP posterior estimate)
        mu = self.gp.predict(self.y, theta_test, return_cov=False, return_var=False)

        # Always add flat prior to keep it in bounds
        mu += self._lnprior(theta_test)

        # Catch NaNs/Infs because they can (rarely) happen
        if not np.isfinite(mu):
            return -np.inf
        else:
            return mu
    # end function


    def run(self, theta=None, y=None, m0=20, m=10, M=10000, nmax=2, Dmax=0.1,
            kmax=5, sampler=None, sim_annealing=False, cv=None, seed=None,
            which_kernel="ExpSquaredKernel", bounds=None, debug=True,
            n_kl_samples=100000, verbose=True, **kw):
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
        cv : int (optional)
            If not None, cv is the number (k) of k-folds CV to use.  Defaults to
            None (no CV)
        seed : int (optional)
            RNG seed.  Defaults to None.
        bounds : tuple/iterable (optional)
            Bounds for minimization scheme.  See scipy.optimize.minimize details
            for more information.  Defaults to None.
        debug : bool (optional)
            Output/plot diagnostic stats/figures?  Defaults to True.
        n_kl_samples : int (optionals)
            Number of samples to draw for Monte Carlo approximation to KL
            divergence between current and previous estimate of the posterior.
            Defaults to 10000.  Error on estimation decreases as approximately
            1/sqrt(n_kl_samples).
        verbose : bool (optional)
            Output all the diagnostics? Defaults to True.

        Returns
        -------
        None
        """

        # Choose m0 initial design points to initialize dataset if none are
        # given
        if theta is None:
            theta = self.prior_sample(m0)
        else:
            theta = np.array(theta)

        if y is None:
            y = self._lnprob(theta)
        else:
            y = np.array(y)

        # Setup, optimize gaussian process XXX just using default options now
        self.gp = gp_utils.setup_gp(theta, y, which_kernel=which_kernel)
        self.gp = gp_utils.optimize_gp(self.gp, theta, y, cv=cv, seed=seed,
                                       which_kernel=which_kernel)

        # Store theta, y
        self.theta = theta
        self.y = y

        # Main loop
        for nn in range(nmax):

            # 1) Find m new points by maximizing utility function
            for ii in range(m):
                theta_t = ut.minimize_objective(self.utility, self.y, self.gp,
                                                sample_fn=self.prior_sample,
                                                prior_fn=self._lnprior,
                                                sim_annealing=sim_annealing,
                                                bounds=bounds, **kw)

                # 2) Query oracle at new points, theta_t
                ### XXX Just using loglikelihood + log prior for now
                #y_t = self._lnlike(theta_t) + self._lnprior(theta_t)
                y_t = self._lnlike(theta_t) + self.posterior(theta_t)

                # Join theta, y arrays
                self.theta = np.concatenate([self.theta, theta_t])
                self.y = np.concatenate([self.y, y_t])

                # 3) Init new GP with new points, optimize
                self.gp = gp_utils.setup_gp(self.theta, self.y,
                                            which_kernel=which_kernel)
                self.gp = gp_utils.optimize_gp(self.gp, self.theta, self.y,
                                               cv=cv, seed=seed,
                                               which_kernel=which_kernel)

            # Plot GP debug diagnostics?
            if debug:
                fig, _ = pu.plot_gp(self.gp, self.theta, self.y,
                                    return_type="mean", log=True,
                                    save_plot="gp_mu_iter_%d.png" % nn)
                plt.close(fig)

            # GP updated: run sampler to obtain new posterior conditioned on
            # (theta_n, log(L_t)*p_n). Use emcee to obtain posterior
            ndim = self.theta.shape[-1]
            nwalk = 10 * ndim
            nsteps = M

            # Initial guess (random over prior)
            p0 = [self.prior_sample(1) for j in range(nwalk)]
            params = ["x%d" % jj for jj in range(ndim)]

            # Init emcee sampler
            sampler = emcee.EnsembleSampler(nwalk, ndim, self._sample)
            for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
                if verbose:
                    print("%d/%d" % (i+1, nsteps))
            if verbose:
                print("emcee finished!")

            # Save current sampler object
            self.samplers.append(sampler)

            # Estimate burn-in, save it
            iburn = mcmc_utils.estimate_burnin(sampler, nwalk, nsteps, ndim)
            self.iburns.append(iburn)

            if nn > 4:

                # Plot mcmc posterior distributions?
                if debug:
                    if verbose:
                        print(iburn)
                    fig = corner.corner(sampler.flatchain[iburn:],
                                        quantiles=[0.16, 0.5, 0.84],
                                        plot_contours=False);

                    fig.savefig("posterior_%d.png" % nn)
                    plt.clf()

                # Make new posterior function using a Gaussian Mixure model to
                # approximate the posterior.
                # Fit some GMMs!
                # sklean hates infs, Nans, big numbers, but I probs messed up XXX
                mask = (~np.isnan(sampler.flatchain).any(axis=1)) & (~np.isinf(sampler.flatchain).any(axis=1))

                # Select optimal number of components via minimizing BIC
                bic = []
                lowest_bic = 1.0e10
                best_gmm = None
                gmm = GaussianMixture()
                for n_components in range(1,5):
                    gmm.set_params(**{"n_components" : n_components,
                                   "covariance_type" : "full"})
                    gmm.fit(sampler.flatchain[iburn:])
                    bic.append(gmm.bic(sampler.flatchain[iburn:]))

                    if bic[-1] < lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = gmm

                # Refit GMM with the lowest bic
                GMM = best_gmm
                GMM.fit(sampler.flatchain[iburn:])

                # Plot GMM?
                if debug:
                    fig, _ = pu.plot_GMM_loglike(GMM, self.theta,
                                                 save_plot="GMM_ll_iter_%d.png" % nn)
                    plt.close(fig)

                # Save current GMM model
                self.GMMs.append(GMM)

                # Update posterior estimate
                self.prev_posterior = self.posterior
                self.posterior = GMM.score_samples

                # Estimate KL-divergence between previous and current posterior
                # Only do this after the 1st (0th) iteration!
                if False:#nn > 0:
                    # Sample from last iteration's GMM
                    prev_samples, _ = self.GMMs[-2].sample(n_kl_samples)

                    # Estimate KL divergence!
                    self.Dkl.append(ut.kl_numerical(prev_samples,
                                                   self.prev_posterior,
                                                   self.posterior))
                else:
                    self.Dkl.append(0.0)
