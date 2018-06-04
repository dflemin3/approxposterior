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

    def __init__(self, theta, y, gp, lnprior, lnlike, prior_sample,
                 algorithm="BAPE"):
        """
        Initializer.

        Parameters
        ----------
        theta : array-like
            Input features (n_samples x n_features).  Defaults to None.
        y : array-like
            Input result of forward model (n_samples,). Defaults to None.
        gp : george.GP
            Gaussian Process that learns the likelihood conditioned on forward
            model input-output pairs (theta, y)
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

        if theta is None or y is None:
            raise ValueError("ERROR: must supply both theta and y")

        self.theta = np.array(theta).squeeze()
        self.y = np.array(y).squeeze()
        self.gp = gp
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
            raise ValueError(err_msg)

        # Initial approximate posteriors are the prior
        self.posterior = self._lnprior
        self.prev_posterior = self._lnprior

        # Holders to save GMM fits to posteriors, raw samplers, KL divergences,
        # GPs
        self.Dkl = list()
        self.GMMs = list()
        self.samplers = list()
        self.iburns = list()
        self.gps = list()
    # end function


    def _gpll(self, theta, *args, **kwargs):
        """
        Compute the approximate posterior conditional distibution at a given
        point, theta (the likelihood + prior learned by the GP)

        Parameters
        ----------
        theta : array-like
            Test point to evaluate GP posterior conditional distribution

        Returns
        -------
        mu : float
            Mean of predicted GP conditional posterior estimate at theta
        """

        # Make sure it's the right shape
        theta_test = np.array(theta).reshape(1,-1)

        # Sometimes the input values can be crazy and the GP will blow up
        if not np.isfinite(theta_test).any():
            return -np.inf

        # Mean of predictive distribution conditioned on y (GP posterior estimate)
        try:
            mu = self.gp.predict(self.y, theta_test, return_cov=False,
                                 return_var=False)
        except ValueError:
            return -np.inf

        # Reject point if prior forbids it
        if not np.isfinite(self._lnprior(theta_test.reshape(-1,))):
            return -np.inf

        # Catch NaNs/Infs because they can (rarely) happen
        if not np.isfinite(mu):
            return -np.inf
        else:
            return mu
    # end function


    def run(self, m0=20, m=10, M=10000, nmax=2, Dmax=0.01,
            kmax=5, sampler=None, p0=None, seed=None, timing=False,
            bounds=None, n_kl_samples=100000, verbose=True, initial_metric=None,
            args=None, **kwargs):
        """
        Core algorithm to estimate the posterior distribution via Gaussian
        Process regression to the joint distribution for the forward model
        input/output pairs (in a Bayesian framework, of course!)

        Parameters
        ----------
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
        p0 : array (optional)
            Initial guess for MCMC walkers.  Defaults to None and creates guess
            from priors.
        seed : int (optional)
            RNG seed.  Defaults to None.
        timing : bool (optional)
            Whether or not to time the code for profiling/speed tests.
            Defaults to False.
        bounds : tuple/iterable (optional)
            Bounds for minimization scheme.  See scipy.optimize.minimize details
            for more information.  Defaults to None.
        n_kl_samples : int (optionals)
            Number of samples to draw for Monte Carlo approximation to KL
            divergence between current and previous estimate of the posterior.
            Defaults to 10000.  Error on estimation decreases as approximately
            1/sqrt(n_kl_samples).
        verbose : bool (optional)
            Output all the diagnostics? Defaults to True.
        initial_metric : array (optional)
            Initial guess for the GP metric.  Defaults to None and is estimated to
            be the squared mean of theta.  In general, you should
            provide your own!

        Returns
        -------
        None
        """

        # Make args empty list if not supplied
        if args is None:
            args = list()

        # Create containers for timing?
        if timing:
            self.training_time = list()
            self.mcmc_time = list()
            self.gmm_time = list()
            self.kl_time = list()

        # Optimize gaussian process
        self.gp = gp_utils.optimize_gp(self.gp, self.theta, self.y, seed=seed)

        # Main loop
        kk = 0
        for nn in range(nmax):

            # 1) Find m new points by maximizing utility function, one at a time
            # Not we call a minimizer because minimizing negative of utility
            # function is the same as maximizing it
            if timing:
                start = time.time()
            for ii in range(m):
                theta_t = ut.minimize_objective(self.utility, self.y, self.gp,
                                                sample_fn=self.prior_sample,
                                                prior_fn=self._lnprior,
                                                bounds=bounds, **kwargs)

                theta_t = np.array(theta_t).reshape(-1,)

                # 2) Query forward model at new point, theta_t
                y_t = np.array([self._lnlike(theta_t, *args, **kwargs) + self._lnprior(theta_t)])

                # If y_t isn't finite, you're likelihood function is messed up
                err_msg = "ERROR: Non-finite likelihood, forward model probably returning NaNs. y_t: %e" % y_t
                assert np.isfinite(y_t), err_msg

                # Join theta, y arrays with new points
                self.theta = np.concatenate([self.theta, theta_t.reshape(1,-1)])
                self.y = np.concatenate([self.y, y_t])

                # 3) Re-optimize GP with new point, optimize

                # Re-initialize, optimize GP since self.theta's shape changed
                self.gp = gp_utils.setup_gp(self.theta, self.y, self.gp)
                self.gp = gp_utils.optimize_gp(self.gp, self.theta, self.y,
                                               seed=seed)

            if timing:
                self.training_time.append(time.time() - start)

            # GP updated: run sampler to obtain new posterior conditioned on
            # {theta_n, log(L_t*prior)}. Use emcee to obtain posterior

            if timing:
                start = time.time()

            # Initialize emcee sampler if None provided
            #if sampler is None:
            ndim = self.theta.shape[-1]
            nwalk = 10 * ndim
            nsteps = M

            # Create sampler using GP ll function as forward model surrogate
            sampler = emcee.EnsembleSampler(nwalk, ndim, self._gpll, args=args, **kwargs)

            # Sample given, call reset to clear it and prepare it for a new run
            """else:
                # Reset sampler attributes, make lnprob function the gp_ll
                sampler.reset()
                sampler.lnprobfn = self.__gpll

                # Get MCMC parameters
                ndim = sampler.dim # ndims
                nwalk = sampler.k # number of walkers
                nsteps = M
            """

            # Provide initial guess (random over prior) if None provided
            if p0 is None:
                p0 = [self.prior_sample(1) for j in range(nwalk)]

            # Run MCMC!
            for ii, result in enumerate(sampler.sample(p0, iterations=nsteps)):
                if verbose:
                    print("%d/%d" % (ii+1, nsteps))
            if verbose:
                print("emcee finished!")

            # Save current sampler object
            self.samplers.append(sampler)

            # Estimate burn-in, save it
            iburn = mcmc_utils.estimate_burnin(sampler, nwalk, nsteps, ndim)
            self.iburns.append(iburn)

            if timing:
                self.mcmc_time.append(time.time() - start)

            if timing:
                start = time.time()

            # Approximate posterior distribution using a Gaussian Mixure model
            GMM = gmm_utils.fit_gmm(sampler, iburn, max_comp=6, cov_type="full",
                                    use_bic=True)

            if timing:
                self.gmm_time.append(time.time() - start)

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
        # end function


    def forecast(self, theta=None, y=None, m0=20, m=10, seed=None,
                which_kernel="ExpSquaredKernel", bounds=None, verbose=True,
                **kwargs):
        """
        XXX: Broken, needs to be fixed!

        Predict where m new forward models should be ran in parameter space
        given input (theta) output (y) pairs (or use the ones we already have!).
        If theta and y are not given, simulate m0 (theta, y) pairs, then
        train a GP to make the prediction.  This is pretty much kriging, aka
        GP regression, but with a predictive step using the utility function.

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
        seed : int (optional)
            RNG seed.  Defaults to None.
        which_kernel : str (optional)
            Which george kernel to use.  Defaults to ExpSquaredKernel.
        bounds : tuple/iterable (optional)
            Bounds for minimization scheme.  See scipy.optimize.minimize details
            for more information.  Defaults to None.
        verbose : bool (optional)
            Output all the diagnostics? Defaults to True.

        Returns
        -------
        theta_hat : array
            New points in parameter space shape: (m, n_dim)
        """

        raise NotImplementedError("dflemin3 broke this and needs to fix it!")

        # Containers for new points
        theta_hat = list()

        # If no input output pair is given, simulate m0 using the
        # forward model or use ones object already has:
        if theta is None or y is None:
            # If we don't have stored values, simulate new ones
            if self.theta is None or self.y is None:
                theta = self.prior_sample(m0)
                y = self._lnlike(theta, *args, **kwargs) + self._lnprior(theta)
            # Have stored values, use a copy
            else:
                theta = self.theta.copy()
                y = self.y.copy()
        # Better be numpy arrays
        else:
            theta = np.array(theta)
            y = np.array(y)

        # Initialize a GP
        gp = gp_utils.optimize_gp(gp, theta, y, seed=seed)

        # Find m new points
        for ii in range(m):
            theta_t = ut.minimize_objective(self.utility, y, gp,
                                            sample_fn=self.prior_sample,
                                            prior_fn=self._lnprior,
                                            bounds=bounds, **kwargs)

            # Save new values
            theta_hat.append(theta_t)

        # Done! return new points
        return np.array(theta_hat).squeeze()
    # end function
