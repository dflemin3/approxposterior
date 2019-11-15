# -*- coding: utf-8 -*-
"""
:py:mod:`approx.py` - ApproxPosterior
-------------------------------------

Bayesian Posterior estimation leveraging Dan Forman-Mackey's Gaussian Process
implementation, george, and his Metropolis-Hastings MCMC ensemble sampler
implementation, emcee, and both Wang & Li (2017) and Kandasamy et al. (2015).

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
import george
import os
import warnings


class ApproxPosterior(object):
    """
    Class to approximate the posterior distributions using either the
    Bayesian Active Posterior Estimation (BAPE) by Kandasamy et al. (2015) or
    the AGP (Adaptive Gaussian Process) by Wang & Li (2017).
    """

    def __init__(self, theta, y, lnprior, lnlike, priorSample, bounds, gp=None,
                 algorithm="BAPE"):
        """
        Initializer.

        Parameters
        ----------
        theta : array-like
            Input features (n_samples x n_features).  Defaults to None.
        y : array-like
            Input result of forward model (n_samples,). Defaults to None.
        lnprior : function
            Defines the log prior over the input features.
        lnlike : function
            Defines the log likelihood function.  In this function, it is assumed
            that the forward model is evaluated on the input theta and the output
            is used to evaluate the log likelihood.
        priorSample : function
            Method to randomly sample points over region allowed by prior
        bounds : tuple/iterable
            Hard bounds for parameters
        gp : george.GP (optional)
            Gaussian Process that learns the likelihood conditioned on forward
            model input-output pairs (theta, y). It's recommended that users
            specify their own kernel, GP using george. If None is provided, then
            approxposterior initialized a GP with a single ExpSquaredKernel as
            these work well in practice.
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
            raise ValueError("Must supply both theta and y for initial GP training set.")

        # Tidy up the shapes
        self.theta = np.array(theta).squeeze()
        self.y = np.array(y).squeeze()

        # Make sure y, theta are valid floats
        if np.any(~np.isfinite(self.theta)) or np.any(~np.isfinite(self.y)):
            print("theta, y:", theta, y)
            raise ValueError("All theta and y values must be finite!")

        # Ensure bounds has correct shape
        if len(bounds) != self.theta.shape[-1]:
            err_msg = "ERROR: bounds provided but len(bounds) != ndim.\n"
            err_msg += "ndim = %d, len(bounds) = %d" % (self.theta.shape[-1], len(bounds))
            raise ValueError(err_msg)
        else:
            self.bounds = bounds

        # Set required functions, algorithm
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

        # Holders to save quantities of interest
        self.iburns = list()
        self.ithins = list()
        self.backends = list()

        # Only save last sampler object since they can get pretty huge
        self.sampler = None

        # Initialize gaussian process if none provided
        if gp is None:
            print("No GP specified. Initializing GP using ExpSquaredKernel.")
            self.gp = gpUtils.defaultGP(self.theta, self.y)
        else:
            self.gp = gp
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
            mu = self.gp.predict(self.y, np.array(theta).reshape(1,-1),
                                 return_cov=False,
                                 return_var=False)
        except ValueError:
            return -np.inf, np.nan

        # Catch NaNs/Infs because they can (rarely) happen
        if not np.isfinite(mu):
            return -np.inf, np.nan
        else:
            return mu + lnprior, lnprior
    # end function


    def run(self, m=10, nmax=2, Dmax=None, kmax=None, seed=None,
            timing=False, nKLSamples=None, verbose=True, maxComp=3,
            mcmcKwargs=None, samplerKwargs=None, estBurnin=False,
            thinChains=False, runName="apRun", cache=True,
            maxLnLikeRestarts=3, gmmKwargs=None, gpMethod=None, gpOptions=None,
            gpP0=None, optGPEveryN=1, nGPRestarts=1, nMinObjRestarts=5,
            gpCV=None, onlyLastMCMC=False, args=None, **kwargs):
        """
        Core algorithm to estimate the posterior distribution via Gaussian
        Process regression to the joint distribution for the forward model
        input/output pairs

        Parameters
        ----------
        m : int (optional)
            Number of new input features to find each iteration.  Defaults to 10.
        nmax : int (optional)
            Maximum number of iterations.  Defaults to 2.
        seed : int (optional)
            RNG seed.  Defaults to None.
        timing : bool (optional)
            Whether or not to time the code for profiling/speed tests.
            Defaults to False.
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
            heuristic.  Defaults to True. In general, we recommend users
            inspect the chains (note that approxposterior always at least saves
            the last sampler object, or all chains if cache = True) and
            calculate the burnin after the fact to ensure convergence.
        thinChains : bool (optional)
            Whether or not to thin chains before GMM fitting.  Useful if running
            long chains.  Defaults to True.  If true, estimates a thin cadence
            via int(0.5*np.min(tau)) where tau is the intergrated autocorrelation
            time.
        runName : str (optional)
            Filename for hdf5 file where mcmc chains are saved.  Defaults to
            apRun and will be saved as apRunii.h5 for ii in range(nmax).
        cache : bool (optional)
            Whether or not to cache MCMC chains, forward model input-output
            pairs, and GP kernel parameters.  Defaults to True since they're
            expensive to evaluate. In practice, users should cache forward model
            inputs, outputs, ancillary parameters, etc in each likelihood
            function evaluation, but saving theta and y here doesn't hurt.
            Saves the forward model, results to runNameAPFModelCache.npz,
            the chains as runNameii.h5 for each, iteration ii, and the GP
            parameters in runNameAPGP.npz in the current working directory, etc.
        maxLnLikeRestarts : int (optional)
            Number of times to restart loglikelihood function (the one that
            calls the forward model) if the lnlike fn returns infs/NaNs. Defaults
            to 3.
        gpMethod : str (optional)
            scipy.optimize.minimize method used when optimized GP hyperparameters.
            Defaults to None, which is nelder-mead, and it usually works.
        gpOptions : dict (optional)
            kwargs for the scipy.optimize.minimize function used to optimize GP
            hyperparameters.  Defaults to None.
        gpP0 : array (optional)
            Initial guess for kernel hyperparameters.  If None, defaults to
            np.random.randn for each parameter.
        optGPEveryN : int (optional)
            How often to optimize the GP hyperparameters.  Defaults to
            re-optimizing everytime a new design point is found, e.g. every time
            a new (theta, y) pair is added to the training set.  Increase this
            parameter if approxposterior is running slowly.
        nGPRestarts : int (optional)
            Number of times to restart GP hyperparameter optimization.  Defaults
            to 1. Increase this number if the GP isn't optimized well.
        nMinObjRestarts : int (optional)
            Number of times to restart minimizing -utility function to select
            next point to improve GP performance.  Defaults to 5.  Increase this
            number of the point selection is not working well.
        gpCV : int (optional)
            Whether or not to use k-fold cross-validation to select kernel
            hyperparameters from the nGPRestarts maximum likelihood solutions.
            Defaults to None. This can be useful if the GP is overfitting, but
            will likely slow down the code. Defaults to None. If using it, perform
            gpCV-fold cross-validation.
        onlyLastMCMC : bool (optional)
            Whether or not to only run the MCMC last iteration. Defaults to False.
            If true, bypasses all KL divergence and related calculations.
        args : iterable (optional)
            Arguments for user-specified loglikelihood function that calls the
            forward model. Defaults to None.
        kwargs : dict (optional)
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model.

        Returns
        -------
        None
        """

        # KL-divergence based convergence is deprecated - warn user if they
        # use it!
        if Dmax is not None or kmax is not None or nKLSamples is not None or gmmKwargs is not None:
            if verbose:
                warn_msg = "KL-divergence convergence is deprecated in " + \
                "approxposterior version 0.21+. The code will ignore " + \
                "Dmax, kmax, nKLSamples, and gmmKwargs. The algorithm will" + \
                "run for nmax iterations."
                warnings.warn(warn_msg, DeprecationWarning)

        # Save forward model input-output pairs since they take forever to
        # calculate and we want them around in case something weird happens.
        # Users should probably do this in their likelihood function
        # anyways, but might as well do it here too.
        if cache:
            np.savez(str(runName) + "APFModelCache.npz",
                     theta=self.theta, y=self.y)

        # Set RNG seed?
        if seed is not None:
            np.random.seed(seed)

        # Create containers for timing?
        if timing:
            self.trainingTime = list()
            self.mcmcTime = list()

        # Initial optimization of gaussian process
        self.gp = gpUtils.optimizeGP(self.gp, self.theta, self.y, seed=seed,
                                     method=gpMethod, options=gpOptions,
                                     p0=gpP0, nGPRestarts=nGPRestarts,
                                     gpCV=gpCV)

        # Main loop - run for nmax iterations
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

                # Reoptimize GP hyperparameters? Note: always optimize 1st time
                if ii % int(optGPEveryN) == 0:
                    optGP = True
                else:
                    optGP = False

                # Find new (theta, y) pair
                # ComputeLnLike = True means new points are saved in self.theta,
                # and self.y, gradually expanding the training set
                self.findNextPoint(computeLnLike=True,
                                   bounds=self.bounds,
                                   maxLnLikeRestarts=maxLnLikeRestarts,
                                   seed=seed,
                                   cache=cache,
                                   gpMethod=gpMethod,
                                   gpOptions=gpOptions,
                                   optGP=optGP,
                                   nGPRestarts=nGPRestarts,
                                   nMinObjRestarts=nMinObjRestarts,
                                   gpCV=gpCV,
                                   runName=runName,
                                   args=args,
                                   **kwargs)

            if timing:
                self.trainingTime.append(time.time() - start)

            # If cache, save current GP hyperparameters
            if cache:
                np.savez(str(runName) + "APGP.npz",
                         gpParamNames=self.gp.get_parameter_names(),
                         gpParamValues=self.gp.get_parameter_vector())

            # GP updated: run MCMC sampler to obtain new posterior conditioned
            # on {theta_n, log(L_t*prior)}. Use emcee to obtain posterior dist.

            # If user only wants to run the MCMC at the end and it's not the
            # last iteration, skip everything below!
            if onlyLastMCMC and nn != (nmax - 1):

                # No sampler yet...
                self.sampler = None

                # Skip everything below
                continue

            # Run the MCMC
            if timing:
                start = time.time()

            self.sampler, iburn, ithin = self.runMCMC(samplerKwargs=samplerKwargs,
                                                      mcmcKwargs=mcmcKwargs,
                                                      runName=str(runName) + str(nn),
                                                      cache=cache,
                                                      estBurnin=estBurnin,
                                                      thinChains=thinChains,
                                                      verbose=verbose,
                                                      args=args,
                                                      kwargs=kwargs)

            # Save burn-in, thin estimates
            self.iburns.append(iburn)
            self.ithins.append(ithin)

            if timing:
                self.mcmcTime.append(time.time() - start)

            # Save timing information?
            if cache:
                if timing:
                    np.savez(str(runName) + "APTiming.npz",
                             trainingTime=self.trainingTime,
                             mcmcTime=self.mcmcTime)
    # end function


    def findNextPoint(self, computeLnLike=True, bounds=None, gpMethod=None,
                      maxLnLikeRestarts=3, seed=None, cache=True, gpOptions=None,
                      gpP0=None, optGP=True, args=None, nGPRestarts=1,
                      nMinObjRestarts=5, gpCV=None, runName="apRun",
                      **kwargs):
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
            the lnlikelihood and lnprior. Defaults to True.
        bounds : tuple/iterable (optional)
            Bounds for minimization scheme.  See scipy.optimize.minimize details
            for more information.  Defaults to None, but it's typically good to
            provide them to ensure a valid solution.
        maxLnLikeRestarts : int (optional)
            Number of times to restart loglikelihood function (the one that
            calls the forward model) if the lnlike fn returns infs/NaNs. Defaults
            to 3.
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
            Defaults to None, which is powell, and it usually works.
        gpOptions : dict (optional)
            kwargs for the scipy.optimize.minimize function used to optimize GP
            hyperparameters.  Defaults to None.
        gpP0 : array (optional)
            Initial guess for kernel hyperparameters.  If None, defaults to
            np.random.randn for each parameter.
        optGP : bool (optional)
            Whether or not to optimize the GP hyperparameters.  Defaults to
            True.
        nGPRestarts : int (optional)
            Number of times to restart GP hyperparameter optimization.  Defaults
            to 1. Increase this number if the GP isn't optimized well.
        nMinObjRestarts : int (optional)
            Number of times to restart minimizing -utility function to select
            next point to improve GP performance.  Defaults to 5.  Increase this
            number of the point selection is not working well.
        gpCV : int (optional)
            Whether or not to use 5-fold cross-validation to select kernel
            hyperparameters from the nGPRestarts maximum likelihood solutions.
            Defaults to None. This can be useful if the GP is overfitting, but
            will likely slow down the code.
        runName : str (optional)
            Filename for hdf5 file where mcmc chains are saved.  Defaults to
            apRun and will be saved as apRunii.h5 for ii in range(nmax).
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
                                          nMinObjRestarts=nMinObjRestarts)

            # Compute lnLikelihood at thetaT?
            if computeLnLike:
                # 2) Query forward model at new point, thetaT
                # Evaluate forward model via loglikelihood function
                loglikeT = self._lnlike(thetaT, *args, **kwargs)

                # If loglike function returns loglike, blobs, ..., only use loglike
                if hasattr(loglikeT, "__iter__"):
                    #yT = np.array([loglikeT[0] + self._lnprior(thetaT)])
                    yT = loglikeT[0]
                else:
                    #yT = np.array([loglikeT + self._lnprior(thetaT)])
                    yT = loglikeT

            # Don't compute lnlikelihood, found point, so we're done
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
                # Create GP using same kernel, updated estimate of the mean, but new theta
                currentHype = self.gp.get_parameter_vector()
                self.gp = george.GP(kernel=self.gp.kernel, fit_mean=True,
                                    mean=np.median(self.y),
                                    white_noise=self.gp.white_noise,
                                    fit_white_noise=False)
                self.gp.set_parameter_vector(currentHype)
                self.gp.compute(self.theta)
                # Now optimize GP given new points?
                if optGP:
                    self.gp = gpUtils.optimizeGP(self.gp, self.theta, self.y,
                                                 seed=seed, method=gpMethod,
                                                 options=gpOptions, p0=gpP0,
                                                 nGPRestarts=nGPRestarts,
                                                 gpCV=gpCV)
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
                # If scaling, save theta in physical units
                np.savez(str(runName)+"APFModelCache.npz",
                         theta=self.theta, y=self.y)
        # Don't care about lnlikelihood, just return thetaT
        return thetaT
    # end function


    def runMCMC(self, samplerKwargs=None, mcmcKwargs=None, runName="apRun",
                cache=True, estBurnin=True, thinChains=True, verbose=False,
                args=None, **kwargs):
        """
        Given forward model input-output pairs, theta and y, and a trained GP,
        run an MCMC using the GP to evaluate the logprobability required by
        MCMC.

        Parameters
        ----------
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
        runName : str (optional)
            Filename prefix for all cached files, e.g. for hdf5 file where mcmc
            chains are saved.  Defaults to runNameii.h5. where ii is the
            current iteration number.
        cache : bool (optional)
            Whether or not to cache MCMC chains, forward model input-output
            pairs, and GP kernel parameters.  Defaults to True since they're
            expensive to evaluate. In practice, users should cache forward model
            inputs, outputs, ancillary parameters, etc in each likelihood
            function evaluation, but saving theta and y here doesn't hurt.
            Saves the forward model, results to runNameAPFModelCache.npz,
            the chains as runNameii.h5 for each, iteration ii, and the GP
            parameters in runNameAPGP.npz in the current working directory, etc.
        estBurnin : bool (optional)
            Estimate burn-in time using integrated autocorrelation time
            heuristic.  Defaults to True. In general, we recommend users
            inspect the chains and calculate the burnin after the fact to ensure
            convergence.
        thinChains : bool (optional)
            Whether or not to thin chains before GMM fitting.  Useful if running
            long chains.  Defaults to True.  If true, estimates a thin cadence
            via int(0.5*np.min(tau)) where tau is the intergrated autocorrelation
            time.
        verbose : bool (optional)
            Output all the diagnostics? Defaults to False.
        args : iterable (optional)
            Arguments for user-specified loglikelihood function that calls the
            forward model. Defaults to None.
        kwargs : dict (optional)
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model.

        Returns
        -------
        sampler : emcee.EnsembleSampler
            emcee sampler object
        iburn : int
            burn-in index estimate.  If estBurnin == False, returns 0.
        ithin : int
            thin cadence estimate.  If thinChains == False, returns 1.
        """

        # Initialize, validate emcee.EnsembleSampler and run_mcmc parameters
        samplerKwargs, mcmcKwargs = mcmcUtils.validateMCMCKwargs(self,
                                                                 samplerKwargs,
                                                                 mcmcKwargs,
                                                                 verbose)

        # Create backend to save chains?
        if cache:
            bname = str(runName) + ".h5"
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
            iburn = int(2.0*np.max(tau))
        else:
            iburn = 0

        # Thin chains?
        if thinChains:
            ithin = np.max((int(0.5*np.min(tau)), 1))
        else:
            ithin = 1

        if verbose:
            print("burn-in estimate: %d" % iburn)
            print("thin estimate: %d" % ithin)

        return self.sampler, iburn, ithin
    # end function
