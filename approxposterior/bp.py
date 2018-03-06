# -*- coding: utf-8 -*-
"""

Bayesian Posterior estimation routines, written in pure python, leveraging
Dan Forman-Mackey's Gaussian Process implementation, george, and his
Metropolis-Hastings MCMC implementation, emcee. We include hybrid
implementations of both Wang & Li (2017) and Kandasamy et al. (2015).  If you
use this, cite them!

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# Tell module what it's allowed to import
__all__ = ["ApproxPosterior"]

from . import utility as ut
from . import gp_utils
from . import mcmc_utils
from . import plot_utils as pu
from . import gmm_utils

import numpy as np
import time
import emcee
import corner
import matplotlib.pyplot as plt


class ApproxPosterior(object):
    """
    Class to approximate the posterior distributions using either the
    Bayesian Active Posterior Estimation (BAPE) by Kandasamy et al. (2015) or
    the AGP (Adaptive Gaussian Process) by Wang & Li (2017).
    """

    def __init__(self, lnprior, lnlike, prior_sample, algorithm="BAPE"):
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


    def run(self, theta=None, y=None, m0=20, m=10, M=10000, nmax=2, Dmax=0.01,
            kmax=5, sampler=None, cv=None, seed=None, timing=False,
            which_kernel="ExpSquaredKernel", bounds=None, debug=True,
            n_kl_samples=100000, verbose=True, update_prior=False, **kw):
        """
        Core algorithm to estimate the posterior distribution via Gaussian
        Process regression to the joint distribution for the forward model
        input/output pairs (in a Bayesian framework, of course!)

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
        cv : int (optional)
            If not None, cv is the number (k) of k-folds CV to use.  Defaults to
            None (no CV)
        seed : int (optional)
            RNG seed.  Defaults to None.
        timing : bool (optional)
            Whether or not to time the code for profiling/speed tests.
            Defaults to False.
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
        update_prior : bool (optional)
            Update the prior function with the current estimate of the posterior
            following Wang & Li (2017)?  Defaults to False (what BAPE does).

        Returns
        -------
        None
        """

        # Create containers for timing?
        if timing:
            self.training_time = list()
            self.mcmc_time = list()
            self.gmm_time = list()
            self.kl_time = list()

        # Choose m0 initial design points to initialize dataset if none given
        if theta is None:
            theta = self.prior_sample(m0)
        else:
            theta = np.array(theta)

        if y is None:
            y = self._lnlike(theta) + self._lnprior(theta)
        else:
            y = np.array(y)

        # Store quantities
        self.theta = theta
        self.y = y

        # Setup, optimize gaussian process
        self.gp = gp_utils.setup_gp(self.theta, self.y, which_kernel=which_kernel)
        self.gp = gp_utils.optimize_gp(self.gp, theta, self.y, cv=cv, seed=seed,
                                       which_kernel=which_kernel)

        # Main loop
        kk = 0
        for nn in range(nmax):

            # 1) Find m new points by maximizing utility function, one at a time
            if timing:
                start = time.time()
            for ii in range(m):
                theta_t = ut.minimize_objective(self.utility, self.y, self.gp,
                                                sample_fn=self.prior_sample,
                                                prior_fn=self._lnprior,
                                                bounds=bounds, **kw)

                # 2) Query forward model at new point, theta_t
                if update_prior:
                    y_t = self._lnlike(theta_t) + self.posterior(theta_t)
                else:
                    y_t = self._lnlike(theta_t) + self._lnprior(theta_t)

                # Join theta, y arrays with new points
                self.theta = np.concatenate([self.theta, theta_t])
                self.y = np.concatenate([self.y, y_t])

                # 3) Initialize new GP with new point, optimize
                self.gp = gp_utils.setup_gp(self.theta, self.y,
                                            which_kernel=which_kernel)
                self.gp = gp_utils.optimize_gp(self.gp, self.theta, self.y,
                                               cv=cv, seed=seed,
                                               which_kernel=which_kernel)

            if timing:
                self.training_time.append(time.time() - start)

            # Plot GP debug diagnostics?
            if debug:
                fig, _ = pu.plot_gp(self.gp, self.theta, self.y,
                                    return_type="mean", log=True,
                                    save_plot="gp_mu_iter_%d.png" % nn)
                plt.close(fig)

            # GP updated: run sampler to obtain new posterior conditioned on
            # {theta_n, log(L_t*prior)}. Use emcee to obtain posterior
            ndim = self.theta.shape[-1]
            nwalk = 10 * ndim
            nsteps = M

            # Initial guess (random over prior)
            p0 = [self.prior_sample(1) for j in range(nwalk)]

            if timing:
                start = time.time()

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

            if timing:
                self.mcmc_time.append(time.time() - start)

            # Plot mcmc posterior distributions?
            if debug:
                if verbose:
                    print(iburn)
                fig = corner.corner(sampler.flatchain[iburn:],
                                    quantiles=[0.16, 0.5, 0.84],
                                    plot_contours=True);

                fig.savefig("posterior_%d.png" % nn)
                plt.clf()

            if timing:
                start = time.time()

            # Approximate posterior distribution using a Gaussian Mixure model
            GMM = gmm_utils.fit_gmm(sampler, iburn, max_comp=6, cov_type="full",
                                    use_bic=True)

            if timing:
                self.gmm_time.append(time.time() - start)

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

            if timing:
                start = time.time()

            # Estimate KL-divergence between previous and current posterior
            # Only do this after the 1st (0th) iteration!
            if nn > 0:
                # Sample from last iteration's GMM
                prev_samples, _ = self.GMMs[-2].sample(n_kl_samples)

                # Numerically estimate KL divergence
                self.Dkl.append(ut.kl_numerical(prev_samples,
                                               self.prev_posterior,
                                               self.posterior))
            else:
                self.Dkl.append(0.0)

            if timing:
                self.kl_time.append(time.time() - start)

            # Convergence diagnostics: If KL divergence is less than threshold
            # for kmax consecutive iterations, we're finished

            # Can't check for convergence on 1st (0th) iteration
            if nn < 1:
                delta_Dkl = 1.0e10
            else:
                delta_Dkl = np.fabs(self.Dkl[-1] - self.Dkl[-2])

            # If the KL divergence is too large, reset counter
            if delta_Dkl <= Dmax:
                kk = kk + 1
            else:
                kk = 0

            # Have we converged?
            if kk >= kmax:
                if verbose:
                    print("Converged! n_iters, Dkl, Delta Dkl: %d, %e, %e" % (nn,self.Dkl[-1],delta_Dkl))
                return
