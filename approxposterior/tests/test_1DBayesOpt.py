#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test Bayesian optimization of 1D function

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import approx, likelihood as lh, gpUtils
import numpy as np
from scipy.optimize import minimize

def test_1DBO():
    """
    Test Bayesian optimization of 1D function
    """

    # Define algorithm parameters
    m0 = 3                           # Size of initial training set
    bounds = [[-1, 2]]               # Prior bounds
    algorithm = "jones"              # Expected Utility from Jones et al. (1998)
    numNewPoints = 10                # Number of new design points to find
    seed = 57                        # RNG seed
    np.random.seed(seed)

    # First, directly minimize the objective to find the "true minimum"
    fn = lambda x : -(lh.testBOFn(x) + lh.testBOFnLnPrior(x))
    trueSoln = minimize(fn, lh.testBOFnSample(1), method="nelder-mead")

    # Sample design points from prior to create initial training set
    theta = lh.testBOFnSample(m0)

    # Evaluate forward model log likelihood + lnprior for each point
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lh.testBOFn(theta[ii]) + lh.testBOFnLnPrior(theta[ii])

    # Initialize default gp with an ExpSquaredKernel
    gp = gpUtils.defaultGP(theta, y, fitAmp=True)

    # Initialize object using the Wang & Li (2017) Rosenbrock function example
    ap = approx.ApproxPosterior(theta=theta,
                                y=y,
                                gp=gp,
                                lnprior=lh.testBOFnLnPrior,
                                lnlike=lh.testBOFn,
                                priorSample=lh.testBOFnSample,
                                bounds=bounds,
                                algorithm=algorithm)

    # Run the Bayesian optimization!
    soln = ap.bayesOpt(nmax=numNewPoints, tol=1.0e-3, seed=seed, verbose=False,
                       cache=False, gpMethod="powell", optGPEveryN=1,
                       nGPRestarts=3, nMinObjRestarts=5, initGPOpt=True,
                       minObjMethod="nelder-mead", findMAP=True,
                       gpHyperPrior=gpUtils.defaultHyperPrior)

    # Ensure estimated maximum and value are within 5% of the truth
    errMsg = "thetaBest is incorrect."
    assert(np.allclose(soln["thetaBest"], trueSoln["x"], rtol=5.0e-2)), errMsg

    errMsg = "Maximum function value is incorrect."
    assert(np.allclose(soln["valBest"], -trueSoln["fun"], rtol=5.0e-2)), errMsg

    # Same as above, but for MAP solution
    errMsg = "thetaBest is incorrect."
    assert(np.allclose(soln["thetaMAPBest"], trueSoln["x"], rtol=5.0e-2)), errMsg

    errMsg = "Maximum function value is incorrect."
    assert(np.allclose(soln["valMAPBest"], -trueSoln["fun"], rtol=5.0e-2)), errMsg

# end function

if __name__ == "__main__":
    test_1DBO()
