# -*- coding: utf-8 -*-
"""
:py:mod:`approx.py` - ApproxPosterior
-------------------------------------

Bayesian Posterior estimation using Dan Forman-Mackey's Gaussian Process
implementation, george, to learn the logprobability of points and the
Metropolis-Hastings MCMC ensemble sampler, emcee, to infer the approximate
posterior distributions given the GP model. approxposterior uses both
Wang & Li (2017) and Kandasamy et al. (2015) utility functions for point
selection to optimially expand the GP's training set.

"""

# Tell module what it's allowed to import
__all__ = ["ApproxPosterior"]

from . import utility as ut
from . import gpUtils
from . import mcmcUtils

import numpy as np
from scipy.optimize import minimize
import time
import emcee
import george
import os
import warnings


class ApproxPosterior(object):
    """
    Class to approximate Bayesian posterior distributions using the
    Bayesian Active Posterior Estimation (BAPE) by Kandasamy et al. (2015),
    the AGP (Adaptive Gaussian Process) by Wang & Li (2017), or a custom
    naive utility function.
    """

    def __init__(self, theta, y, lnprior, lnlike, priorSample, bounds, gp=None,
                 algorithm="bape"):
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
            Point selection algorithm that specifies which utility function to
            use.  Defaults to bape.  Options are bape,
            agp, alternate (between bape and agp), and naive.
            Case doesn't matter. If alternate, runs agp on even numbers and bape
            on odd.

        Returns
        -------
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
        elif self.algorithm == "naive":
            self.utility = ut.NaiveUtility
        else:
            errMsg = "Unknown algorithm. Valid options: bape, agp, naive, or alternate."
            raise ValueError(errMsg)

        # Holders to save quantities of interest
        self.iburns = list()
        self.ithins = list()
        self.backends = list()

        # Only save last sampler object since they can get pretty huge
        self.sampler = None

        # Initialize gaussian process if none provided
        if gp is None:
            print("INFO: No GP specified. Initializing GP using ExpSquaredKernel.")
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
            return mu, lnprior
    # end function


    def optGP(self, seed=None, method="powell", options=None, p0=None,
              nGPRestarts=1, gpHyperPrior=gpUtils.defaultHyperPrior):
        """
        Optimize hyperparameters of object's GP

        Parameters
        ----------
        seed : int (optional)
            numpy RNG seed.  Defaults to None.
        nGPRestarts : int (optional)
            Number of times to restart the optimization.  Defaults to 1. Increase
            this number if the GP isn't optimized well.
        method : str (optional)
            scipy.optimize.minimize method.  Defaults to powell.
        options : dict (optional)
            kwargs for the scipy.optimize.minimize function.  Defaults to None.
        p0 : array (optional)
            Initial guess for kernel hyperparameters.  If None, defaults to
            np.random.randn for each parameter
        gpHyperPrior : str/callable (optional)
            Prior function for GP hyperparameters. Defaults to the defaultHyperPrior fn.
            This function asserts that the mean must be negative and that each log
            hyperparameter is within the range [-20,20].

        Returns
        -------
        optimizedGP : george.GP
        """

        # Optimize and reasign gp
        self.gp = gpUtils.optimizeGP(self.gp, self.theta, self.y, seed=seed,
                                     method=method, options=options,
                                     p0=p0, nGPRestarts=nGPRestarts,
                                     gpHyperPrior=gpHyperPrior)
    # end function


    def run(self, m=10, nmax=2,seed=None, timing=False, verbose=True,
            mcmcKwargs=None, samplerKwargs=None, estBurnin=False,
            thinChains=False, runName="apRun", cache=True, maxLnLikeRestarts=3,
            gpMethod="powell", gpOptions=None, gpP0=None, optGPEveryN=1,
            nGPRestarts=1, nMinObjRestarts=5, onlyLastMCMC=False,
            initGPOpt=True, gpHyperPrior=gpUtils.defaultHyperPrior,
            dropInitialTraining=False, args=None, **kwargs):
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
            to 3. If you find the need to increase this parameter, don't because
            your lnlike or lnprior function is probably at fault or your model
            is not properly specified.
        gpMethod : str (optional)
            scipy.optimize.minimize method used when optimized GP hyperparameters.
            Defaults to powell (it usually works)
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
        onlyLastMCMC : bool (optional)
            Whether or not to only run the MCMC last iteration. Defaults to False.
        initGPOpt : bool (optional)
            Whether or not to optimize GP hyperparameters before 0th iteration.
            Defaults to True (aka assume user didn't optimize GP hyperparameters)
        gpHyperPrior : str/callable (optional)
            Prior function for GP hyperparameters. Defaults to the defaultHyperPrior fn.
            This function asserts that the mean must be negative and that each log
            hyperparameter is within the range [-20,20].
        dropInitialTraining : bool (optional)
            Whether or not to drop the initial training set and only regress
            the GP on points it chose after learning on the initial training set.
            This can be useful in cases where approxposterior uses the initial
            training set to identify the high likelihood regions while not needing
            to regress on low-likelihood/useless points in the initial training
            set. Defaults to False.
        args : iterable (optional)
            Arguments for user-specified loglikelihood function that calls the
            forward model. Defaults to None.
        kwargs : dict (optional)
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model.

        Returns
        -------
        """

        # If dropping initial training set after 1st round of point selection,
        # save length on initial training set
        if dropInitialTraining:
            lenToDrop = len(self.y)

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

        # Initial optimization of gaussian process?
        if initGPOpt:
            self.optGP(seed=seed, method=gpMethod, options=gpOptions, p0=gpP0,
                       nGPRestarts=nGPRestarts, gpHyperPrior=gpHyperPrior)

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
                    bOptGP = True
                else:
                    bOptGP = False

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
                                   bOptGP=bOptGP,
                                   nGPRestarts=nGPRestarts,
                                   nMinObjRestarts=nMinObjRestarts,
                                   gpHyperPrior=gpHyperPrior,
                                   runName=runName,
                                   args=args,
                                   **kwargs)

            # Drop the initial training set after 1st round of point selection?
            if dropInitialTraining and nn == 0:
                self.theta = self.theta[lenToDrop:,:]
                self.y = self.y[lenToDrop:]

                # Create GP using same kernel, updated estimate of the mean, but new theta
                currentHype = self.gp.get_parameter_vector()
                self.gp = george.GP(kernel=self.gp.kernel, fit_mean=True,
                                    mean=self.gp.mean,
                                    white_noise=self.gp.white_noise,
                                    fit_white_noise=False)
                self.gp.set_parameter_vector(currentHype)
                self.gp.compute(self.theta)

                self.optGP(seed=seed, method=gpMethod, options=gpOptions,
                           p0=gpP0, nGPRestarts=nGPRestarts,
                           gpHyperPrior=gpHyperPrior)

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

            if timing:
                start = time.time()

            # Run the MCMC
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
                      gpP0=None, bOptGP=True, args=None, nGPRestarts=1,
                      nMinObjRestarts=5, runName="apRun",
                      gpHyperPrior=gpUtils.defaultHyperPrior, **kwargs):
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
            apFModelCache.npz in the current working directory (name can change
            if user specifies runName).
        gpMethod : str (optional)
            scipy.optimize.minimize method used when optimized GP hyperparameters.
            Defaults to None, which is powell, and it usually works.
        gpOptions : dict (optional)
            kwargs for the scipy.optimize.minimize function used to optimize GP
            hyperparameters.  Defaults to None.
        gpP0 : array (optional)
            Initial guess for kernel hyperparameters.  If None, defaults to
            np.random.randn for each parameter.
        bOptGP : bool (optional)
            Whether or not to optimize the GP hyperparameters.  Defaults to
            True.
        nGPRestarts : int (optional)
            Number of times to restart GP hyperparameter optimization.  Defaults
            to 1. Increase this number if the GP isn't optimized well.
        nMinObjRestarts : int (optional)
            Number of times to restart minimizing -utility function to select
            next point to improve GP performance.  Defaults to 5.  Increase this
            number of the point selection is not working well.
        runName : str (optional)
            Filename for hdf5 file where mcmc chains are saved.  Defaults to
            apRun.
        gpHyperPrior : str/callable (optional)
            Prior function for GP hyperparameters. Defaults to the defaultHyperPrior fn.
            This function asserts that the mean must be negative and that each log
            hyperparameter is within the range [-20,20].
        args : iterable (optional)
            Arguments for user-specified loglikelihood function that calls the
            forward model. Defaults to None.
        kwargs : dict (optional)
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model. Defaults to None.

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
                    yT = np.array([loglikeT[0] + self._lnprior(thetaT)])
                else:
                    yT = np.array([loglikeT + self._lnprior(thetaT)])

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
                                    mean=self.gp.mean,
                                    white_noise=self.gp.white_noise,
                                    fit_white_noise=False)
                self.gp.set_parameter_vector(currentHype)
                self.gp.compute(self.theta)
                # Now optimize GP given new points?
                if bOptGP:
                    self.optGP(seed=seed, method=gpMethod, options=gpOptions,
                               p0=gpP0, nGPRestarts=nGPRestarts,
                               gpHyperPrior=gpHyperPrior)
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
        run an MCMC using the GP to evaluate the logprobability instead of the
        true, computationally-expensive forward model.

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
            convergence, but this function works pretty well.
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


    def findMAP(self, theta0=None, method="nelder-mead", options=None,
                nRestarts=5):
        """
        Find maximum a posteriori (MAP) estimate, given a trained GP. To find
        the MAP, this function minimizes -mean predicted by the GP, aka finds
        what the GP believes is the point of maximum logprobability.

        Parameters
        ----------
        theta0 : iterable
            Initial guess. Defaults to a sample from the prior function.
        method : str (optional)
            scipy.optimize.minimize method.  Defaults to powell.
        options : dict (optional)
            kwargs for the scipy.optimize.minimize function.  Defaults to None.
        nRestarts : int (optional)
            Number of times to restart the optimization. Defaults to 5.

        Returns
        -------
        MAP : iterable
            maximum a posteriori estimate
        fn : float
            Mean of GP predictive function at MAP solution
        """

        # Initialize theta0 if not provided. If provided, validate it
        if theta0 is not None:
            theta0 = np.array(theta0).squeeze()
            assert theta0.shape == theta.shape[-1]

        # Figure out if we can supply bounds
        if str(method).lower() in ["l-bfgs-b", "tnc"]:
            bounds = self.bounds
        else:
            bounds = None

        # Initialize option if method is nelder-mead and options not provided
        if str(method.lower()) == "nelder-mead":
            if options is None:
                options = {"adaptive" : True}

        # Containers for solutions
        res = []
        vals = []

        # Set optimization fn for MAP
        def fn(x):
            # If not allowed by the prior, reject!
            if not np.isfinite(self._lnprior(x)):
                return np.inf
            else:
                return -(self._gpll(x)[0])

        # Loop over optimization calls
        for ii in range(nRestarts):
            # Keep minimizing until a valid solution is found
            while True:
                # Guess initial point
                if theta0 is None:
                    t0 = self.theta[np.argmax(self.y)] + 1.0e-3 * np.random.randn()
                else:
                    # Perturb user-supplied guess
                    t0 = np.array(theta0) + np.min(theta0) * 1.0e-3 * np.random.randn(len(theta0))

                tmp = minimize(fn, t0, method=method, options=options,
                               bounds=bounds)["x"]

                # If solution is finite and allowed by the prior, save!
                if np.all(np.isfinite(tmp)):
                    if np.isfinite(self._lnprior(tmp)):
                        # Save solution, function value
                        res.append(tmp)
                        vals.append(fn(tmp))
                        break

        # Return best answer
        bestInd = np.argmin(vals)

        return res[bestInd], vals[bestInd]
    # end function
# end class
