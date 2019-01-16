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
import os
from scipy.optimize import minimize


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
            Which utility function to use.  Defaults to BAPE.  Options are BAPE,
            AGP, or alternate.  Case doesn't matter. If alternate, runs AGP on
            even numbers and BAPE on odd.

        Returns
        -------
        None
        """

        # Need to supply the training set
        if theta is None or y is None:
            raise ValueError("Must supply both theta and y")

        self.theta = np.array(theta).squeeze()
        self.y = np.array(y).squeeze()

        # Make sure y, theta are valid floats
        if np.any(~np.isfinite(self.theta)) or np.any(~np.isfinite(self.y)):
            print("theta, y:", theta, y)
            raise ValueError("Both theta and y must all be finite!")

        self.gp = gp
        self._lnprior = lnprior
        self._lnlike = lnlike
        self.priorSample = priorSample
        self.algorithm = str(algorithm).lower()

        # Assign utility function
        if self.algorithm == "bape":
            self.utility = ut.BAPEUtility
        elif self.algorithm == "agp":
            self.utility = ut.AGPUtility
        elif self.algorithm == "alternate":
            # If alternate, AGP on even, BAPE on odd
            self.utility = ut.AGPUtility
        else:
            errMsg = "Unknown algorithm. Valid options: BAPE, AGP, or alternate."
            raise ValueError(errMsg)

        # Initial approximate posteriors are the prior
        self.posterior = self._lnprior
        self.prevPosterior = self._lnprior

        # Holders to save GMM fits to posteriors, raw samplers, KL divergences
        self.Dkl = list()
        self.GMMs = list()
        self.iburns = list()
        self.ithins = list()
        self.backends = list()

        # Only save last sampler object since they can get pretty huge
        self.sampler = None
    # end function


    def _gpll(self, theta, *args, **kwargs):
        """
        Compute the approximate posterior conditional distibution, the
        likelihood + prior learned by the GP, at a given point, theta.

        Parameters
        ----------
        theta : array-like
            Test point to evaluate GP posterior conditional distribution

        Returns
        -------
        mu : float
            Mean of predicted GP conditional posterior estimate at theta
        lnprior : float
            log prior evlatuated at theta
        """

        # Sometimes the input values can be crazy and the GP will blow up
        if not np.any(np.isfinite(theta)):
            return -np.inf, np.nan

        # Reject point if prior forbids it
        lnprior = self._lnprior(theta)
        if not np.isfinite(lnprior):
            return -np.inf, np.nan

        # Mean of predictive distribution conditioned on y (GP posterior estimate)
        # and make sure theta is the right shape for the GP
        try:
            mu = self.gp.predict(self.y,
                                 np.array(theta).reshape(1,-1),
                                 return_cov=False,
                                 return_var=False)
        except ValueError:
            return -np.inf, np.nan

        # Catch NaNs/Infs because they can (rarely) happen
        if not np.isfinite(mu):
            return -np.inf, np.nan
        else:
            return mu, lnprior
    # end function


    def run(self, m=10, nmax=2, Dmax=0.01, kmax=5, seed=None,
            timing=False, bounds=None, nKLSamples=10000, verbose=True,
            args=None, maxComp=3, mcmcKwargs=None, samplerKwargs=None,
            estBurnin=False, thinChains=False, chainFile="apRun", cache=True,
            maxLnLikeRestarts=5, gmmKwargs=None, gpMethod=None, gpOptions=None,
            **kwargs):
        """
        Core algorithm to estimate the posterior distribution via Gaussian
        Process regression to the joint distribution for the forward model
        input/output pairs

        Parameters
        ----------
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
        estBurnin : bool (optional)
            Estimate burn-in time using integrated autocorrelation time
            heuristic.  Defaults to False as in general, we recommend users
            inspect the chains and calculate the burnin after the fact to ensure
            convergence.
        thinChains : bool (optional)
            Whether or not to thin chains before GMM fitting.  Useful if running
            long chains.  Defaults to False.  If true, estimates a thin cadence
            via int(0.5*np.min(tau)) where tau is the intergrated autocorrelation
            time.
        chainFile : str (optional)
            Filename for hdf5 file where mcmc chains are saved.  Defaults to
            apRun and will be saved as apRunii.h5 for ii in nmax.
        cache : bool (optional)
            Whether or not to cache MCMC chains and forward model input-output
            pairs.  Defaults to True since the both are expensive to evaluate.
            In practice, users should cache forward model inputs, outputs,
            ancillary parameters, etc in each likelihood function evaluation,
            but saving theta and y here doesn't hurt.  Saves the forward model
            results to apFModelCache.npz and the chains as apRunii.h5 for each
            iteration ii in the current working directory.
        maxLnLikeRestarts : int (optional)
            Number of times to restart loglikelihood function (the one that
            calls the forward model) if the lnlike fn returns infs/NaNs. Defaults
            to 5.
        gmmKwargs : dict (optional)
            keyword arguments for sklearn.mixture.GaussianMixture. Defaults to
            None
        args : iterable (optional)
            Arguments for user-specified loglikelihood function that calls the
            forward model. Defaults to None.
        gpMethod : str (optional)
            scipy.optimize.minimize method used when optimized GP hyperparameters.
            Defaults to None, which is nelder-mead, and it usually works.
        gpOptions : dict (optional)
            kwargs for the scipy.optimize.minimize function used to optimize GP
            hyperparameters.  Defaults to None.
        kwargs : dict (optional)
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model.

        Returns
        -------
        None
        """

        # Save forward model input-output pairs since they take forever to
        # calculate and we want them around in case something weird happens.
        # Users should probably do this in their likelihood function
        # anyways, but might as well do it here too.
        if cache:
            np.savez("apFModelCache.npz", theta=self.theta, y=self.y)

        # Set RNG seed?
        if seed is not None:
            np.random.seed(seed)

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

        # If scipy.minimize bounds are provided, make sure it has ndim elements
        if bounds is not None and (len(bounds) != samplerKwargs["ndim"]):
            err_msg = "ERROR: bounds provided but len(bounds) != ndim.\n"
            err_msg += "ndim = %d, len(bounds) = %d" % (samplerKwargs["ndim"], len(bounds))
            raise ValueError(err_msg)

        # Initial optimization of gaussian process
        self.gp = gpUtils.optimizeGP(self.gp, self.theta, self.y, seed=seed,
                                     method=gpMethod, options=gpOptions)

        # Main loop
        kk = 0
        for nn in range(nmax):
            if verbose:
                print("Iteration: %d" % nn)

            if timing:
                start = time.time()

            # 1) Find m new points by maximizing utility function, one at a time
            # Note that we call a minimizer because minimizing negative of
            # utility function is the same as maximizing it
            for ii in range(m):

                # If alternating utility functions, switch here!
                if self.algorithm == "alternate":
                    # AGP on even, BAPE on odd
                    if ii % 2 == 0:
                        self.utility = ut.AGPUtility
                    else:
                        self.utility = ut.BAPEUtility

                # computeLnLike=True means new points are saved in self.theta,
                # and self.y
                self.findNextPoint(computeLnLike=True,
                                   bounds=bounds,
                                   maxLnLikeRestarts=maxLnLikeRestarts,
                                   seed=seed,
                                   cache=cache,
                                   args=args,
                                   **kwargs)

            if timing:
                self.trainingTime.append(time.time() - start)

            # GP updated: run sampler to obtain new posterior conditioned on
            # {theta_n, log(L_t*prior)}. Use emcee to obtain posterior

            if timing:
                start = time.time()

            # Create backend to save chains
            if cache:
                bname = chainFile + str(nn) + ".h5"
                self.backends.append(bname)
                backend = emcee.backends.HDFBackend(bname)
                backend.reset(samplerKwargs["nwalkers"], samplerKwargs["ndim"])
            # Only keep last sampler object in memory
            else:
                backend = None

            # Create sampler using GP lnlike function as forward model surrogate
            self.sampler = emcee.EnsembleSampler(**samplerKwargs,
                                                 backend=backend,
                                                 args=args,
                                                 kwargs=kwargs,
                                                 blobs_dtype=[("lnprior", float)])

            # Run MCMC!
            for _ in self.sampler.sample(**mcmcKwargs):
                pass
            if verbose:
                print("mcmc finished")

            # If estimating burn in or thin scale, compute integrated
            # autocorrelation length of the chains
            if estBurnin or thinChains:
                # tol = 0 so it always returns an answer
                tau = self.sampler.get_autocorr_time(tol=0)

                # Catch NaNs
                if np.any(~np.isfinite(tau)):
                    # Try removing NaNs
                    tau = tau[np.isfinite(np.array(tau))]
                    if len(tau) < 1:
                        if verbose:
                            print("Failed to compute integrated autocorrelation length, tau.")
                            print("Setting tau = 1")
                        tau = 1

            # Estimate burn-in?
            if estBurnin:
                # Note we set tol=0 so it always provides an estimate, even if
                # the estimate isn't good, in which case run longer chains!
                iburn = int(2.0*np.max(tau))
            else:
                iburn = 0

            # Thin chains?
            if thinChains:
                ithin = int(0.5*np.min(tau))
            else:
                ithin = 1

            if verbose:
                print("burn-in estimate: %d" % iburn)
                print("thin estimate: %d" % ithin)
            self.iburns.append(iburn)
            self.ithins.append(ithin)

            if timing:
                self.mcmcTime.append(time.time() - start)

            if timing:
                start = time.time()

            # Fit for the approximate posterior distribution using a Gaussian
            # Mixure model
            GMM = gmmUtils.fitGMM(self.sampler.get_chain(discard=iburn, flat=True, thin=ithin),
                                  maxComp=maxComp,
                                  covType="full",
                                  useBic=True,
                                  gmmKwargs=gmmKwargs)

            if verbose:
                print("GMM fit complete")

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
                deltaDkl = np.inf
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


    def findNextPoint(self, computeLnLike=True, bounds=None, gpMethod=None,
                      maxLnLikeRestarts=1, seed=None, cache=True, gpOptions=None,
                      args=None, **kwargs):
        """
        Find new point, thetaT, by maximizing utility function. Note that we
        call a minimizer because minimizing negative of utility function is
        the same as maximizing it.

        This function can be used in 2 ways:
            1) Finding the new point, thetaT, that would maximally improve the
               GP's predictive ability.  This point could be used to select
               where to run a new forward model, for example.
            2) Find a new thetaT and evaluate the forward model at this location
               to iteratively improve the GP's predictive performance, a core
               function of the BAPE and AGP algorithms.

        If computeLnLike is True, all results of this function are appended to
        the corresponding object elements, e.g. thetaT appended to self.theta.
        thetaT is returned, as well as yT if computeLnLike is True.  Note that
        returning yT requires running the forward model and updating the GP.

        Parameters
        ----------
        computeLnLike : bool (optional)
            Whether or not to run the forward model and compute yT, the sum of
            the lnlikelihood and lnprior
        bounds : tuple/iterable (optional)
            Bounds for minimization scheme.  See scipy.optimize.minimize details
            for more information.  Defaults to None.
        maxLnLikeRestarts : int (optional)
            Number of times to restart loglikelihood function (the one that
            calls the forward model) if the lnlike fn returns infs/NaNs. Defaults
            to 5.
        seed : int (optional)
            RNG seed.  Defaults to None.
        cache : bool (optional)
            Whether or not to cache forward model input-output pairs.  Defaults
            to True since the forward model is expensive to evaluate. In
            practice, users should cache forward model inputs, outputs,
            ancillary parameters, etc in each likelihood function evaluation,
            but saving theta and y here doesn't hurt.  Saves the results to
            apFModelCache.npz in the current working directory.
        gpMethod : str (optional)
            scipy.optimize.minimize method used when optimized GP hyperparameters.
            Defaults to None, which is nelder-mead, and it usually works.
        gpOptions : dict (optional)
            kwargs for the scipy.optimize.minimize function used to optimize GP
            hyperparameters.  Defaults to None.
        args : iterable (optional)
            Arguments for user-specified loglikelihood function that calls the
            forward model. Defaults to None.
        kwargs : dict (optional)
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model.

        Returns
        -------
        None
            Returns nothing if computeLnLike is True, as everything is
            saved in self.theta, self.y, and potentially cached if cache is True.
            Otherwise, returns thetaT and yT.
        thetaT : array
            New design point selected by maximizing GP utility function.
        yT : array
            Value of loglikelihood + logprior at thetaT.
        """

        if args is None:
            args = ()

        yT = np.nan
        llIters = 0

        # Find new theta that produces a valid loglikelihood
        while not np.isfinite(yT):

            # If loglikeT isn't finite after maxLnLikeRestarts tries,
            # your likelihood function is not executing properly
            if llIters >= maxLnLikeRestarts:
                errMsg = "Non-finite likelihood for %d iterations." % maxLnLikeRestarts
                errMsg += "Forward model probably returning NaNs."
                print("Last thetaT, loglikeT, yT:", thetaT, loglikeT, yT)
                raise RuntimeError(errMsg)

            # Find new point
            thetaT = ut.minimizeObjective(self.utility, self.y, self.gp,
                                          sampleFn=self.priorSample,
                                          priorFn=self._lnprior,
                                          bounds=bounds, **kwargs)

            # Compute lnLikelihood at thetaT?
            if computeLnLike:
                # 2) Query forward model at new point, thetaT
                # Evaluate forward model via loglikelihood function
                loglikeT = self._lnlike(thetaT, *args, **kwargs)

                # If loglike function returns loglike, blobs, ..., only use loglike
                if hasattr(loglikeT, "__iter__"):
                    yT = np.array([loglikeT[0] + self._lnprior(thetaT)])
                else:
                    yT = np.array([loglikeT + self._lnprior(thetaT)])

            # Don't compute lnlikelihood
            else:
                break

            llIters += 1

        if computeLnLike:
            # Valid theta, y found. Join theta, y arrays with new points.
            self.theta = np.vstack([self.theta, np.array(thetaT)])
            self.y = np.hstack([self.y, yT])

            # 3) Re-optimize GP with new point, optimize

            # Re-initialize, optimize GP since self.theta's shape changed
            try:
                self.gp = gpUtils.setupGP(self.theta, self.y, self.gp)
                self.gp = gpUtils.optimizeGP(self.gp, self.theta, self.y,
                                             seed=seed, method=gpMethod,
                                             options=gpOptions)
            except ValueError:
                print("theta:", self.theta)
                print("y:", self.y)
                print("gp parameters names:", self.gp.get_parameter_names())
                print("gp parameters:", self.gp.get_parameter_vector())
                raise ValueError("GP couldn't optimize!")

            # Save forward model input-output pairs since they take forever to
            # calculate and we want them around in case something weird happens.
            # Users should probably do this in their likelihood function
            # anyways, but might as well do it here too.
            if cache:
                np.savez("apFModelCache.npz", theta=self.theta, y=self.y)
        # Don't care about lnlikelihood, just return thetaT.
        else:
            return thetaT
    # end function


    def findMLE(self, x0, method="nelder-mead", bounds=None, options=None,
                *args, **kwargs):
        """
        Find the maximum likelihood solution according to the GP.

        Parameters
        ----------
        x0 : array
            Initial guess
        method: str (optional)
            minimizer method.  Defaults to nelder-mead.
        bounds : iterable (optional)
            bounds for optimizer
        options : dict (optional)
            kwargs for the minimizer

        Returns
        -------
        thetaHat : array
            Maximum likelihood parameter values
        yHat : float
            Maximal likelihood value
        """

        # Guess better be correct shape
        err_msg = "Initial guess must have same number of dimensions as theta!"
        assert len(x0) == self.theta.shape[-1], err_msg

        # Make sure GP is properly setup
        if self.gp.computed:
            pass
        else:
            raise RuntimeError("ERROR: Need to compute GP before using it!")

        def fn(x, *args, **kwargs):
            """Dummy function to return -loglikelihood function"""
            return -self._gpll(x, *args, **kwargs)[0]

        # Take extra precautions to make sure thetaHat is a valid number
        # since GPs can get a little weird
        thetaHat = np.inf
        while not np.all(np.isfinite(thetaHat)):
            try:
                thetaHat = minimize(fn, x0, method=method, bounds=bounds,
                                    options=options)["x"]
            except ValueError:
                thetaHat = np.inf

            # Vet answer: must be finite, allowed by prior
            # Are all values finite?
            if np.all(np.isfinite(thetaHat)):
                # Is this point allowed by the prior?
                if np.isfinite(self._lnprior(thetaHat, **kwargs)):
                    break

        # Now compute maximum likelihood value at MLE
        yHat = self._gpll(thetaHat)[0]

        return thetaHat, yHat
    # end function
