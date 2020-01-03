# -*- coding: utf-8 -*-
"""
:py:mod:`approx.py` - ApproxPosterior
-------------------------------------

Approximate Bayesian Posterior estimation and Bayesian optimzation. approxposterior
uses Dan Forman-Mackey's Gaussian Process implementation, george, and the
Metropolis-Hastings MCMC ensemble sampler, emcee, to infer the approximate
posterior distributions given the GP model.

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
    Class used to estimate approximate Bayesian posterior distributions or
    perform Bayesian optimization using a Gaussian process surrogate model
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
            Point selection algorithm that specifies which utility (also
            referred to as acquisition) function to use.  Defaults to bape.
            Options are bape (Bayesian Active Learning for Posterior Estimation,
            Kandasamy et al. (2015)), agp (Adapted Gaussian Process Approximation,
            Wang & Li (2017)), alternate (between AGP and BAPE), and jones
            (Jones et al. (1998) expected improvement).
            Case doesn't matter. If alternate, runs agp on even numbers and bape
            on odd.

            For approximate Bayesian posterior estimation, bape or alternate
            are typically the best optimizations. For Bayesian optimization,
            jones (expected improvement) usually performs best.

        Returns
        -------
        """

        # Need to supply the training set
        if theta is None or y is None:
            raise ValueError("Must supply both theta and y for initial GP training set.")

        # Tidy up the shapes
        self.theta = np.array(theta).squeeze()
        self.y = np.array(y).squeeze()

        # Determine dimensionality
        if self.theta.ndim <= 1:
            ndim = 1
        else:
            ndim = theta.shape[-1]

        # Make sure y, theta are valid floats
        if np.any(~np.isfinite(self.theta)) or np.any(~np.isfinite(self.y)):
            print("theta, y:", theta, y)
            raise ValueError("All theta and y values must be finite!")

        # Ensure bounds has correct shape
        if len(bounds) != ndim:
            err_msg = "ERROR: bounds provided but len(bounds) != ndim.\n"
            err_msg += "ndim = %d, len(bounds) = %d" % (ndim, len(bounds))
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
        elif self.algorithm == "jones":
            self.utility = ut.JonesUtility
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
        Optimize hyperparameters of approx object's GP

        Parameters
        ----------
        seed : int (optional)
            numpy RNG seed.  Defaults to None.
        nGPRestarts : int (optional)
            Number of times to restart GP hyperparameter optimization.  Defaults
            to 1. Increase this number if the GP is not well-optimized.
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


    def run(self, m=10, nmax=2, seed=None, timing=False, verbose=True,
            mcmcKwargs=None, samplerKwargs=None, estBurnin=False,
            thinChains=False, runName="apRun", cache=True, gpMethod="powell",
            gpOptions=None, gpP0=None, optGPEveryN=1, nGPRestarts=1,
            nMinObjRestarts=5, onlyLastMCMC=False, initGPOpt=True, kmax=3,
            gpHyperPrior=gpUtils.defaultHyperPrior, eps=1.0, convergenceCheck=False,
            minObjMethod="nelder-mead", minObjOptions=None, args=None, **kwargs):
        """
        Core method to estimate the approximate posterior distribution via
        Gaussian Process regression

        Parameters
        ----------
        m : int (optional)
            Number of new design points to find each iteration. These are the
            points that are selected by maximizing the utility function, e.g.
            bape or agp, and sequentially added to the GP training set.  Defaults
            to 10.
        nmax : int (optional)
            Maximum number of iterations.  Defaults to 2. Algorithm will terminate
            if either nmax iterations is met or the convergence criterion is met
            if convergenceCheck is True.
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
            to 1. Increase this number if the GP is not well-optimized.
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
        eps : float (optional)
            Change in the mean of the approximate marginal posterior
            distributions, relative to the previous marginal posterior distribution's
            standard deviation (aka relative z score), for kmax iterations
            required for convergence. Defaults to 1.0.
        kmax : int (optional)
            Number of consecutive iterations for convergence check to pass before
            successfully ending algorithm. Defaults to 3.
        convergenceCheck : bool (optional)
            Whether or not to terminate the execution if the change in the mean
            of the approximate marginal posterior distributions, relative to the
            previous marginal posterior distribution's standard deviation
            (aka relative z score) varies by less than eps for kmax consecutive
            iterations. Defaults to False. Note: if using this, make sure you're
            confortable with the burnin and thinning applied to the MCMC chains.
            See estBurnin and thinChains parameters.
        minObjMethod : str (optional)
            scipy.optimize.minimize method used when optimizing
            utility functions for point selection.  Defaults to nelder-mead.
        minObjOptions : dict (optional)
            kwargs for the scipy.optimize.minimize function used when optimizing
            utility functions for point selection.  Defaults to None,
            but if method == "nelder-mead", options = {"adaptive" : True}
        args : iterable (optional)
            Arguments for user-specified loglikelihood function that calls the
            forward model. Defaults to None.
        kwargs : dict (optional)
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model.

        Returns
        -------
        """

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

        # Initialize convergence check counter
        kk = 0

        # If checking for convergence, must run the MCMC each iteration
        if convergenceCheck and onlyLastMCMC:
            errMsg = "If convergenceCheck is True, must run an MCMC each iteration.\n"
            errMsg += "convergenceCheck = %d onlyLastMCMC = %d" % (convergenceCheck, onlyLastMCMC)
            raise RuntimeError(errMsg)

        # Main loop - run for nmax iterations
        for nn in range(nmax):
            if verbose:
                print("Iteration: %d" % nn)

            if timing:
                start = time.time()

            # 1) Find m new (theta, y) pairs by maximizing utility function,
            # one at a time. Note that computeLnLike = True means new points are
            # saved in self.theta, and self.y, expanding the training set
            # 2) In this function, GP hyperparameters are reoptimized after every
            # optGPEveryN new points
            _, _ = self.findNextPoint(computeLnLike=True,
                                      seed=seed,
                                      cache=cache,
                                      gpMethod=gpMethod,
                                      gpOptions=gpOptions,
                                      nGPRestarts=nGPRestarts,
                                      nMinObjRestarts=nMinObjRestarts,
                                      optGPEveryN=optGPEveryN,
                                      numNewPoints=m,
                                      gpHyperPrior=gpHyperPrior,
                                      minObjMethod=minObjMethod,
                                      minObjOptions=minObjOptions,
                                      runName=runName,
                                      theta0=None, # Sample from prior
                                      args=args,
                                      **kwargs)

            if timing:
                self.trainingTime.append(time.time() - start)

            # If cache, save current GP hyperparameters
            if cache:
                np.savez(str(runName) + "APGP.npz",
                         gpParamNames=self.gp.get_parameter_names(),
                         gpParamValues=self.gp.get_parameter_vector())

            # 3) GP updated: run MCMC sampler to obtain new posterior conditioned
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

            # Run the MCMC using the trained GP to predict the logprobability
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

            # Convergence check?
            if convergenceCheck:

                # Extract current posterior marginal means, stds
                samples = self.sampler.get_chain(discard=self.iburns[-1],
                                                 flat=True,
                                                 thin=self.ithins[-1])
                meanNN = np.mean(samples, axis=0)
                stdNN = np.std(samples, axis=0)

                # Cannot converge after just one iteration
                if nn == 0:
                    meanPrev = meanNN
                    stdPrev = stdNN
                else:
                    # Compute z score for each parameter mean relative to
                    # previous approximate marginal posterior distribution quantities
                    zScore = np.fabs((meanNN - meanPrev)/stdPrev)
                    if np.all(zScore < eps):
                        kk += 1
                    else:
                        kk = 0

                    # Save previous values
                    meanPrev = meanNN
                    stdPrev = stdNN

                # If close for kmax consecutive iterations, converged!
                if kk >= kmax:
                    if verbose:
                        print("Approximate marginal posterior distributions converged.")
                        print("Delta zScore threshold, eps: %e" % eps)
                        print("kk, kmax: %d, %d" % (kk, kmax))
                        print("Final abs(zScore):", zScore)
                        break
    # end function


    def findNextPoint(self, theta0=None, computeLnLike=True, seed=None,
                      cache=True, gpOptions=None, gpP0=None, args=None,
                      nGPRestarts=1, nMinObjRestarts=5, gpMethod="powell",
                      minObjMethod="nelder-mead", minObjOptions=None,
                      runName="apRun", numNewPoints=1, optGPEveryN=1,
                      gpHyperPrior=gpUtils.defaultHyperPrior, **kwargs):
        """
        Find numNewPoints new point(s), thetaT, by maximizing utility function.
        Note that we call a minimizer because minimizing negative of utility
        function is the same as maximizing it.

        This function can be used in 2 ways:
            1) Finding the new point(s), thetaT, that would maximally improve the
               GP's predictive ability.  This point could be used to select
               where to run a new forward model, for example.
            2) Find a new thetaT and evaluate the forward model at this location
               to iteratively improve the GP's predictive performance, a core
               function of the BAPE and AGP algorithms.

        If computeLnLike is True, all results of this function are appended to
        the corresponding object elements, e.g. thetaT appended to self.theta.
        thetaT is returned, as well as yT if computeLnLike is True.  Note that
        returning yT requires running the forward model and updating the GP.

        If numNewPoints > 1, iteratively find numNewPoints. After each new
        point is found, re-compute the GP covariance matrix. The GP
        hyperparameters are then optionally re-optimized at the specified
        cadence.

        Parameters
        ----------
        theta0 : float/iterable (optional)
            Initial guess for optimization. Defaults to None, which draws a sample
            from the prior function using sampleFn.
        computeLnLike : bool (optional)
            Whether or not to run the forward model and compute yT, the sum of
            the lnlikelihood and lnprior. Defaults to True. If True, also
            appends all new values to self.theta, self.y, in addition to
            returning the new values
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
        optGPEveryN : int (optional)
            How often to optimize the GP hyperparameters.  Defaults to
            re-optimizing everytime a new design point is found, e.g. every time
            a new (theta, y) pair is added to the training set.  Increase this
            parameter if approxposterior is running slowly. NB: GP hyperparameters
            are optimized *only* if computeLnLike == True
        gpMethod : str (optional)
            scipy.optimize.minimize method used when optimized GP hyperparameters.
            Defaults to None, which is powell, and it usually works.
        gpOptions : dict (optional)
            kwargs for the scipy.optimize.minimize function used to optimize GP
            hyperparameters.  Defaults to None.
        gpP0 : array (optional)
            Initial guess for kernel hyperparameters.  If None, defaults to
            np.random.randn for each parameter.
        nGPRestarts : int (optional)
            Number of times to restart GP hyperparameter optimization.  Defaults
            to 1. Increase this number if the GP is not well-optimized.
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
        numNewPoints : int (optional)
            Number of new points to find. Defaults to 1.
        minObjMethod : str (optional)
            scipy.optimize.minimize method used when optimizing
            utility functions for point selection.  Defaults to nelder-mead.
        minObjOptions : dict (optional)
            kwargs for the scipy.optimize.minimize function used when optimizing
            utility functions for point selection.  Defaults to None,
            but if method == "nelder-mead", options = {"adaptive" : True}
        args : iterable (optional)
            Arguments for user-specified loglikelihood function that calls the
            forward model. Defaults to None.
        kwargs : dict (optional)
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model. Defaults to None.

        Returns
        -------
        thetaT : float or iterable
            New design point(s) selected by maximizing GP utility function.
        yT : float or iterable (optional)
            Value(s) of loglikelihood + logprior at thetaT. Only returned if
            computeLnLike == True
        """

        # Validate inputs
        assert (isinstance(numNewPoints, int) and (numNewPoints >= 1))
        assert (isinstance(optGPEveryN, int) and (optGPEveryN >= 1))

        if args is None:
            args = ()

        # Containers for new points
        newTheta = list()
        if computeLnLike:
            newY = list()

        # Find numNewPoints new design points
        for ii in range(numNewPoints):

            # If alternating utility functions, switch here!
            if self.algorithm == "alternate":
                # AGP on even, BAPE on odd
                if ii % 2 == 0:
                    self.utility = ut.AGPUtility
                else:
                    self.utility = ut.BAPEUtility

            # Find new theta that produces a valid loglikelihood
            thetaT, uT = ut.minimizeObjective(self.utility, self.y, self.gp,
                                              sampleFn=self.priorSample,
                                              priorFn=self._lnprior,
                                              nRestarts=nMinObjRestarts,
                                              method=minObjMethod,
                                              options=minObjOptions,
                                              bounds=self.bounds,
                                              theta0=theta0,
                                              args=(self.y,self.gp,self._lnprior))

            # Save new thetaT
            newTheta.append(thetaT)

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

                # Save new logprobability
                newY.append(yT)

                # Valid theta, y found. Join theta, y arrays with new points.
                if self.theta.ndim > 1:
                    self.theta = np.vstack([self.theta, np.array(thetaT)])
                else:
                    self.theta = np.hstack([self.theta, thetaT])
                self.y = np.hstack([self.y, yT])

                # Re-optimize GP with new point, optimize

                # Re-conpute GP's covariance matrix since self.theta's shape changed
                try:
                    # Create GP using same kernel, updated estimate of the mean, but new theta
                    # Always need to compute the covariance matrix when we find
                    # a new theta
                    currentHype = self.gp.get_parameter_vector()
                    self.gp = george.GP(kernel=self.gp.kernel, fit_mean=True,
                                        mean=self.gp.mean,
                                        white_noise=self.gp.white_noise,
                                        fit_white_noise=False)
                    self.gp.set_parameter_vector(currentHype)
                    self.gp.compute(self.theta)

                    # Reoptimize GP hyperparameters?
                    if ii % optGPEveryN == 0:
                        self.optGP(seed=seed, method=gpMethod, options=gpOptions,
                                   p0=gpP0, nGPRestarts=nGPRestarts,
                                   gpHyperPrior=gpHyperPrior)
                    else:
                        pass

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
        # end loop

        # Figure out what to return
        if numNewPoints == 1:
            newTheta = newTheta[0]
            if computeLnLike:
                newY = newY[0]

        if computeLnLike:
            return np.asarray(newTheta), np.asarray(newY)
        else:
            return np.asarray(newTheta)
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
                nRestarts=15):
        """
        Find the maximum a posteriori (MAP) estimate of the function learned
        by the GP. To find the MAP, this function minimizes -mean predicted by
        the GP, aka finds what the GP believes is the maximum of whatever
        function is definded by self._lnlike + self._lnprior.

        Note: MAP estimation typically work better when fitAmp = True, that is
        the GP kernel fits for an amplitude term.

        Parameters
        ----------
        theta0 : iterable
            Initial guess. Defaults to a sample from the prior function.
        method : str (optional)
            scipy.optimize.minimize method.  Defaults to powell.
        options : dict (optional)
            kwargs for the scipy.optimize.minimize function.  Defaults to None.
        nRestarts : int (optional)
            Number of times to restart the optimization. Defaults to 15.

        Returns
        -------
        MAP : iterable
            maximum a posteriori estimate
        MAPVal : float
            Mean of GP predictive function at MAP solution
        """

        # Initialize theta0 if not provided. If provided, validate it
        if theta0 is not None:
            theta0 = np.array(theta0).reshape(1,self.theta.shape[-1])
        else:
            # Guess current max of y
            theta0 = self.theta[np.argmax(self.y)]

        # Initialize option if method is nelder-mead and options not provided
        if str(method).lower() == "nelder-mead":
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

        # Minimize values predicted by GP, i.e. find minimum of mean of GP's
        # conditional posterior distribution
        MAP, MAPVal = ut.minimizeObjective(fn, self.y, self.gp,
                                           self.priorSample, self._lnprior,
                                           nRestarts=nRestarts, args=None,
                                           method=method, options=options,
                                           bounds=self.bounds, theta0=theta0)

        # Return best answer (-MAPVal since we minimized function)
        return MAP, -MAPVal
    # end function


    def bayesOpt(self, nmax, theta0=None, tol=1.0e-3, kmax=3, seed=None,
                 verbose=True, runName="apRun", cache=True, gpMethod="powell",
                 gpOptions=None, gpP0=None, optGPEveryN=1, nGPRestarts=1,
                 nMinObjRestarts=5, initGPOpt=True, minObjMethod="nelder-mead",
                 gpHyperPrior=gpUtils.defaultHyperPrior,  minObjOptions=None,
                 findMAP=True, args=None, **kwargs):
        """
        Perform Bayesian optimization given a GP surrogate model to estimate

        thetaBest = argmax(fn(theta))

        given a GP trained on (theta, y). In this case, fn is the function
        specified by self._lnlike + self._lnprior. Note that this function
        *maximizes* the objective, so if performing a minimization,
        define the objective as the negative of your function. See Brochu et al.
        (2009) or Frazier (2018) for good reviews of Bayesian optimization.

        This function terminates once nmax points have been selected or when
        the function value changes by less than tol over consecutive iterations,
        whichever one happens first.

        Note 1: lnlike does not have to be a log likelihood, but rather can be any
        continous function one wishes to optimize. The function lnprior is used
        to place priors on parameters of the function, theta. The typical use
        of lnprior is to ensure the solution remains within a hypercube or
        simplex, i.e., bounding the possible values of theta.

        Note 2: For this function, it is recommended to keep optGPEveryN = 1 to
        ensure the GP properly learns the underlying function.

        Note 3: Bayesian optimization and MAP estimation typically work better
        when fitAmp = True, that is the GP kernel has an amplitude term

        Parameters
        ----------
        nmax : int
            Maximum number of new design points to find. These are the
            points that are selected by maximizing the utility function, e.g.
            the expected improvement, and sequentially added to the GP training
            set.
        theta0 : iterable
            Initial guess. Defaults to a sample from the prior function.
        tol : float (optional)
            Convergence tolerance. This function will terminate if the function
            value at the estimated extremum changes by less than tol over
            kmax consecutive iterations. Defaults to 1.0e-3.
        kmax : int (optional)
            Number of iterations required for the difference in estimated
            extremum functions values < tol required for convergence. Defaults
            to 3.
        seed : int (optional)
            RNG seed.  Defaults to None.
        verbose : bool (optional)
            Output all the diagnostics? Defaults to True.
        runName : str (optional)
            Filename to prepend to cache files where model input-output pairs
            and the current GP hyperparameter values are saved. Defaults to
            apRun.
        cache : bool (optional)
            Whether or not to cache forward model input-output pairs, and GP
            kernel parameters.  Defaults to True since they're
            expensive to evaluate. In practice, users should cache forward model
            inputs, outputs, ancillary parameters, etc in each likelihood
            function evaluation, but saving theta and y here doesn't hurt.
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
            a new (theta, y) pair is added to the training set.
        nGPRestarts : int (optional)
            Number of times to restart GP hyperparameter optimization.  Defaults
            to 1. Increase this number if the GP is not well-optimized.
        nMinObjRestarts : int (optional)
            Number of times to restart minimizing -utility function to select
            next point to improve GP performance.  Defaults to 5.  Increase this
            number of the point selection is not working well.
        initGPOpt : bool (optional)
            Whether or not to optimize GP hyperparameters before 0th iteration.
            Defaults to True (aka assume user didn't optimize GP hyperparameters)
        gpHyperPrior : str/callable (optional)
            Prior function for GP hyperparameters. Defaults to the defaultHyperPrior fn.
            This function asserts that the mean must be negative and that each log
            hyperparameter is within the range [-20,20].
        minObjMethod : str (optional)
            scipy.optimize.minimize method used when optimizing
            utility functions for point selection.  Defaults to nelder-mead.
        minObjOptions : dict (optional)
            kwargs for the scipy.optimize.minimize function used when optimizing
            utility functions for point selection.  Defaults to None,
            but if method == "nelder-mead", options = {"adaptive" : True}
        args : iterable (optional)
            Arguments for user-specified loglikelihood function that calls the
            forward model. Defaults to None.
        kwargs : dict (optional)
            Keyword arguments for user-specified loglikelihood function that
            calls the forward model.

        Returns
        -------
        soln : dict
            Dictionary that contains the following keys: "thetaBest" : Best fit
            solution, "valBest" : function value at best fit solution, thetas :
            solution vector, vals : function values along solution, "nev" :
            number of forward model evaluations, aka number of iterations
        """

        # Define holders for solutions
        thetas = list()
        vals = list()
        if findMAP:
            thetasMAP = list()
            valsMAP = list()

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

        # Initial optimization of gaussian process?
        if initGPOpt:
            self.optGP(seed=seed, method=gpMethod, options=gpOptions, p0=gpP0,
                       nGPRestarts=nGPRestarts, gpHyperPrior=gpHyperPrior)

        # Main loop - run for nmax iterations or until converged
        kk = 0
        for nn in range(nmax):
            if verbose:
                print("Iteration: %d" % nn)

            # Figure out when to re-optimize GP hyperparameters
            if nn % optGPEveryN == 0:
                optN = 1
            else:
                # Huge number so it doesn't reoptimize hyperparameters this iter
                optN = 99999999

            # 1) Find new (theta, y) pairs by maximizing utility (acquisition)
            # function, Note that computeLnLike = True means new points are
            # saved in self.theta, and self.y, expanding the training set
            thetaT, yT = self.findNextPoint(computeLnLike=True,
                                            seed=seed,
                                            cache=cache,
                                            gpMethod=gpMethod,
                                            gpOptions=gpOptions,
                                            nGPRestarts=nGPRestarts,
                                            nMinObjRestarts=nMinObjRestarts,
                                            optGPEveryN=optN,
                                            numNewPoints=1,
                                            gpHyperPrior=gpHyperPrior,
                                            minObjMethod=minObjMethod,
                                            minObjOptions=minObjOptions,
                                            runName=runName,
                                            args=args,
                                            **kwargs)

            if verbose:
                print("Forward model evaluation at: ", thetaT, ", function value: ", yT)

            # If cache, save current GP hyperparameters
            if cache:
                np.savez(str(runName) + "APGP.npz",
                         gpParamNames=self.gp.get_parameter_names(),
                         gpParamValues=self.gp.get_parameter_vector())

            # 3a) Cache current best forward model run
            thetas.append(self.theta[np.argmax(self.y)])
            vals.append(self.y[np.argmax(self.y)])

            # 3b) Find current MAP solution?
            if findMAP:
                thetaN, valN = self.findMAP(theta0=theta0, method=minObjMethod,
                                            options=minObjOptions,
                                            nRestarts=nMinObjRestarts)

                if verbose:
                    print("Current MAP solution: ", thetaN, valN)

                # Save point
                thetasMAP.append(thetaN)
                valsMAP.append(valN)

            # Convergence check
            if nn > 0:
                if np.fabs(vals[-1] - vals[-2]) < tol:
                    kk = kk + 1
                # Not close enough: reset counter
                else:
                    kk = 0

                # Converged for enough consecutive iterations?
                if kk >= kmax:
                    break
        # end loop

        # Create solution dictionary that is sort of like minimizer's
        # OptimizerSolution object
        soln = {"thetaBest" : thetas[-1], "valBest" : vals[-1],
                "thetas" : np.asarray(thetas).squeeze(),
                "vals" : np.asarray(vals).squeeze(), "nev" : nn+1}

        if findMAP:
            soln["thetasMAP"] = np.asarray(thetasMAP).squeeze()
            soln["valsMAP"] = np.asarray(valsMAP).squeeze()
            soln["thetaMAPBest"] = soln["thetasMAP"][np.argmax(soln["valsMAP"])]
            soln["valMAPBest"] = soln["valsMAP"][np.argmax(soln["valsMAP"])]

        return soln
    # end function

# end class
