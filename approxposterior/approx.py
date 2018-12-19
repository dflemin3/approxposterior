# -*- coding: utf-8 -*-
"""

Bayesian Posterior estimation routines, written in pure python, leveraging
Dan Forman-Mackey's Gaussian Process implementation, george, and his
Metropolis-Hastings MCMC implementation, emcee. We include hybrid
implementations of both Wang & Li (2017) and Kandasamy et al. (2015).  If you
use this, cite them!

"""

# Tell module what it's allowed to import
__all__ = ["ApproxPosterior"]

from . import utility as ut
from . import gpUtils
from . import mcmcUtils
from . import gmmUtils

import numpy as np
import time
import emcee


class ApproxPosterior(object):
    """
    Class to approximate the posterior distributions using either the
    Bayesian Active Posterior Estimation (BAPE) by Kandasamy et al. (2015) or
    the AGP (Adaptive Gaussian Process) by Wang & Li (2017).
    """

    def __init__(self, theta, y, gp, lnprior, lnlike, priorSample,
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
        priorSample : function
            Method to randomly sample points over region allowed by prior
        algorithm : str (optional)
            Which utility function to use.  Defaults to BAPE.  Options are BAPE
            or AGP.  Case doesn't matter.

        Returns
        -------
        None
        """

        # Need to supply the training set
        if theta is None or y is None:
            raise ValueError("ERROR: must supply both theta and y")

        self.theta = np.array(theta).squeeze()
        self.y = np.array(y).squeeze()
        self.gp = gp
        self._lnprior = lnprior
        self._lnlike = lnlike
        self.priorSample = priorSample
        self.algorithm = algorithm

        # Assign utility function
        if self.algorithm.lower() == "bape":
            self.utility = ut.BAPEUtility
        elif self.algorithm.lower() == "agp":
            self.utility = ut.AGPUtility
        else:
            errMsg = "ERROR: Invalid algorithm. Valid options: BAPE, AGP."
            raise ValueError(errMsg)

        # Initial approximate posteriors are the prior
        self.posterior = self._lnprior
        self.prevPosterior = self._lnprior

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
        thetaTest = np.array(theta).reshape(1,-1)

        # Sometimes the input values can be crazy and the GP will blow up
        if not np.isfinite(thetaTest).any():
            return -np.inf

        # Reject point if prior forbids it
        if not np.isfinite(self._lnprior(thetaTest.reshape(-1,))):
            return -np.inf

        # Mean of predictive distribution conditioned on y (GP posterior estimate)
        try:
            mu = self.gp.predict(self.y, thetaTest, return_cov=False,
                                 return_var=False)
        except ValueError:
            return -np.inf

        # Catch NaNs/Infs because they can (rarely) happen
        if not np.isfinite(mu):
            return -np.inf
        else:
            return mu
    # end function


    def run(self, m0=20, m=10, nmax=2, Dmax=0.01, kmax=5, seed=None,
            timing=False, bounds=None, nKLSamples=100000, verbose=True,
            args=None, maxComp=3, mcmcKwargs=None, samplerKwargs=None,
            estBurnin=False, **kwargs):
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
        nmax : int (optional)
            Maximum number of iterations.  Defaults to 2 for testing.
        Dmax : float (optional)
            Maximum change in KL divergence for convergence checking.  Defaults to 0.1.
        kmax : int (optional)
            Maximum number of iterators such that if the change in KL divergence is
            less than Dmax for kmax iterators, the algorithm is considered
            converged and terminates.  Defaults to 5.
        seed : int (optional)
            RNG seed.  Defaults to None.
        timing : bool (optional)
            Whether or not to time the code for profiling/speed tests.
            Defaults to False.
        bounds : tuple/iterable (optional)
            Bounds for minimization scheme.  See scipy.optimize.minimize details
            for more information.  Defaults to None.
        nKLSamples : int (optionals)
            Number of samples to draw for Monte Carlo approximation to KL
            divergence between current and previous estimate of the posterior.
            Defaults to 10000.  Error on estimation decreases as approximately
            1/sqrt(nKLSamples).
        verbose : bool (optional)
            Output all the diagnostics? Defaults to True.
        maxComp : int (optional)
            Maximum number of mixture model components to fit for when fitting a
            GMM model to approximate the posterior distribution.  Defaults to 3.
        samplerKwargs : dict (optional)
            Parameters for emcee.EnsembleSampler object
            If None, defaults to the following:
                nwalkers : int (optional)
                    Number of emcee walkers.  Defaults to 10 * dim
        mcmcKwargs : dict (optional)
            Parameters for emcee.EnsembleSampler.sample/.run_mcmc methods. If
            None, defaults to the following required parameters:
                iterations : int (optional)
                    Number of MCMC steps.  Defaults to 10,000
                initial_state : array/emcee.State (optional)
                    Initial guess for MCMC walkers.  Defaults to None and
                    creates guess from priors.
        args : iterable (optional)
            Arguments for user-specified loglikelihood function that calls the
            forward model.
        estBurnin : bool (optional)
            Estimate burn-in time using integrated autocorrelation time
            heuristic.  Defaults to False as in general, we recommend users
            inspect the chains and calculate the burnin after the fact to ensure
            convergence.
        kwargs : dict (optional)
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model.


        Returns
        -------
        None
        """

        # Make args empty list if not supplied
        if args is None:
            args = list()

        # Create containers for timing?
        if timing:
            self.trainingTime = list()
            self.mcmcTime = list()
            self.gmmTime = list()
            self.klTime = list()

        # Initialize, validate emcee.EnsembleSampler and run_mcmc parameters
        samplerKwargs["ndim"] = self.theta.shape[-1]
        samplerKwargs, mcmcKwargs = mcmcUtils.validateMCMCKwargs(samplerKwargs,
                                                                 mcmcKwargs,
                                                                 self,
                                                                 verbose)

        # Inital optimization of gaussian process
        self.gp = gpUtils.optimizeGP(self.gp, self.theta, self.y, seed=seed)

        # Main loop
        kk = 0
        for nn in range(nmax):
            if verbose:
                print("Iteration: %d" % nn)

            # 1) Find m new points by maximizing utility function, one at a time
            # Note that we call a minimizer because minimizing negative of
            # utility function is the same as maximizing it
            if timing:
                start = time.time()
            for ii in range(m):
                thetaT = ut.minimizeObjective(self.utility, self.y, self.gp,
                                              sampleFn=self.priorSample,
                                              priorFn=self._lnprior,
                                              bounds=bounds, **kwargs)

                # I'm very particular about my shapes
                thetaT = np.array(thetaT).reshape(-1,)

                # 2) Query forward model at new point, thetaT
                ### TODO: try multiprocessing here for lnlike to speed it up?

                # Evaluate forward model via loglikelihood function
                loglikeT = self._lnlike(thetaT, *args, **kwargs)

                # If loglike function returns loglike, blobs, ..., only use loglike
                if hasattr(loglikeT, "__iter__"):
                    yT = np.array([loglikeT[0] + self._lnprior(thetaT)])
                else:
                    yT = np.array([loglikeT + self._lnprior(thetaT)])

                # If yT isn't finite, your likelihood function is messed up
                errMsg = "ERROR: Non-finite likelihood, forward model probably returning NaNs. yT: %e" % yT
                assert np.isfinite(yT), errMsg

                # Join theta, y arrays with new points
                self.theta = np.concatenate([self.theta, thetaT.reshape(1,-1)])
                self.y = np.concatenate([self.y, yT])

                # 3) Re-optimize GP with new point, optimize

                # Re-initialize, optimize GP since self.theta's shape changed
                self.gp = gpUtils.setupGP(self.theta, self.y, self.gp)
                self.gp = gpUtils.optimizeGP(self.gp, self.theta, self.y,
                                               seed=seed)

            if timing:
                self.trainingTime.append(time.time() - start)

            # GP updated: run sampler to obtain new posterior conditioned on
            # {theta_n, log(L_t*prior)}. Use emcee to obtain posterior

            if timing:
                start = time.time()

            # Create sampler using GP lnlike function as forward model surrogate
            sampler = emcee.EnsembleSampler(**samplerKwargs, args=args,
                                            kwargs=kwargs)

            # Run MCMC!
            for ii, result in enumerate(sampler.sample(**mcmcKwargs)):
                if verbose:
                    print("%d/%d" % (ii+1, mcmcKwargs["iterations"]))
            if verbose:
                print("emcee finished!")

            # Save current sampler object
            self.samplers.append(sampler)

            # Estimate burn-in, save it
            if estBurnin:
                # Note we set tol=0 so it always provides an estimate, even if
                # the estimate isn't good.  If estimate isn't good, run longer
                # chains!
                iburn = int(2.0*np.max(sampler.get_autocorr_time(tol=0)))
            # Don't estimate burnin, keep all chains
            else:
                iburn = 0

            if verbose:
                print("burnin estimate: %d" % iburn)
            self.iburns.append(iburn)

            if timing:
                self.mcmcTime.append(time.time() - start)

            if timing:
                start = time.time()

            # Approximate posterior distribution using a Gaussian Mixure model
            GMM = gmmUtils.fitGMM(sampler.flatchain[iburn:], maxComp=maxComp,
                                  covType="full", useBic=True)

            if verbose:
                print("GMM fit.")

            if timing:
                self.gmmTime.append(time.time() - start)

            # Save current GMM model
            self.GMMs.append(GMM)

            # Update posterior estimate
            self.prevPosterior = self.posterior
            self.posterior = GMM.score_samples

            if timing:
                start = time.time()

            # Estimate KL-divergence between previous and current posterior
            # Only do this after the 1st (0th) iteration!
            if nn > 0:
                # Sample from last iteration's GMM
                prevSamples, _ = self.GMMs[-2].sample(nKLSamples)

                # Numerically estimate KL divergence
                self.Dkl.append(ut.klNumerical(prevSamples,
                                               self.prevPosterior,
                                               self.posterior))
            else:
                self.Dkl.append(0.0)

            if timing:
                self.klTime.append(time.time() - start)

            # Convergence diagnostics: If KL divergence is less than threshold
            # for kmax consecutive iterations, we're finished

            # Can't check for convergence on 1st (0th) iteration
            if nn < 1:
                deltaDkl = 1.0e10
            else:
                deltaDkl = np.fabs(self.Dkl[-1] - self.Dkl[-2])

            # If the KL divergence is too large, reset counter
            if deltaDkl <= Dmax:
                kk = kk + 1
            else:
                kk = 0

            # Have we converged?
            if kk >= kmax:
                if verbose:
                    print("Converged! n_iters, Dkl, Delta Dkl: %d, %e, %e" % (nn,self.Dkl[-1],deltaDkl))
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
        thetaHat : array
            New points in parameter space shape: (m, n_dim)
        """

        raise NotImplementedError("dflemin3 broke this and needs to fix it!")

        # Containers for new points
        thetaHat = list()

        # If no input output pair is given, simulate m0 using the
        # forward model or use ones object already has:
        if theta is None or y is None:
            # If we don't have stored values, simulate new ones
            if self.theta is None or self.y is None:
                theta = self.priorSample(m0)
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
        gp = gpUtils.optimizeGP(gp, theta, y, seed=seed)

        # Find m new points
        for ii in range(m):
            thetaT = ut.minimizeObjective(self.utility, y, gp,
                                          sampleFn=self.priorSample,
                                          priorFn=self._lnprior,
                                          bounds=bounds, **kwargs)

            # Save new values
            thetaHat.append(thetaT)

        # Done! return new points
        return np.array(thetaHat).squeeze()
    # end function
