#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test Bayesian optimization of 2D Rosenbrock function

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import approx, likelihood as lh, gpUtils
import numpy as np
from scipy.optimize import minimize

def test_2DBO():
    """
    Test Bayesian optimization of 2D Rosenbrock function
    """

    # Define algorithm parameters
    m0 = 10                          # Size of initial training set
    bounds = [[-5, 5], [-5, 5]]      # Prior bounds
    algorithm = "jones"              # Expected Utility from Jones et al. (1998)
    numNewPoints = 10                # Number of new design points to find
    seed = 91                        # RNG seed
    np.random.seed(seed)

    # First, directly minimize the objective
    fn = lambda x : -(lh.sphereLnlike(x) + lh.sphereLnprior(x))
    trueSoln = minimize(fn, lh.sphereSample(1), method="nelder-mead")

    # Sample design points from prior to create initial training set
    theta = lh.sphereSample(m0)

    # Evaluate forward model log likelihood + lnprior for each point
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lh.sphereLnlike(theta[ii]) + lh.sphereLnprior(theta[ii])

    # Initialize default gp with an ExpSquaredKernel
    gp = gpUtils.defaultGP(theta, y, fitAmp=True)

    # Initialize object using the Wang & Li (2017) Rosenbrock function example
    ap = approx.ApproxPosterior(theta=theta,
                                y=y,
                                gp=gp,
                                lnprior=lh.sphereLnprior,
                                lnlike=lh.sphereLnlike,
                                priorSample=lh.sphereSample,
                                bounds=bounds,
                                algorithm=algorithm)

    # Run the Bayesian optimization!
    soln = ap.bayesOpt(nmax=numNewPoints, tol=1.0e-3, kmax=3, seed=seed,
                       cache=False, gpMethod="powell", optGPEveryN=1,
                       nGPRestarts=3, nMinObjRestarts=5, initGPOpt=True,
                       minObjMethod="nelder-mead", verbose=False, findMAP=True,
                       gpHyperPrior=gpUtils.defaultHyperPrior)

    # Ensure estimated maximum and value are within small value of the truth
    errMsg = "thetaBest is incorrect."
    assert(np.allclose(soln["thetaBest"], trueSoln["x"], atol=1.0e-2)), errMsg

    errMsg = "Maximum function value is incorrect."
    assert(np.allclose(soln["valBest"], trueSoln["fun"], atol=1.0e-2)), errMsg

    # Same as above, but for the MAP solution
    errMsg = "MAP thetaBest is incorrect."
    assert(np.allclose(soln["thetaMAPBest"], trueSoln["x"], atol=1.0e-2)), errMsg

    errMsg = "MAP function value is incorrect."
    assert(np.allclose(soln["valMAPBest"], trueSoln["fun"], atol=1.0e-2)), errMsg

# end function

if __name__ == "__main__":
    test_2DBO()
