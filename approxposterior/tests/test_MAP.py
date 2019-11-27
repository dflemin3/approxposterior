#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test finding an MAP estimate.

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import approx, likelihood as lh, gpUtils
import numpy as np


def testMAPAmp():
    """
    Test MAP estimation
    """

    # Define algorithm parameters
    m0 = 50                           # Initial size of training set
    bounds = ((-5,5), (-5,5))         # Prior bounds
    algorithm = "bape"                # Use the Kandasamy et al. (2015) formalism
    seed = 57                         # For reproducibility
    np.random.seed(seed)

    # Randomly sample initial conditions from the prior
    theta = np.array(lh.rosenbrockSample(m0))

    # Evaluate forward model log likelihood + lnprior for each theta
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

    # Create the the default GP using an ExpSquaredKernel
    gp = gpUtils.defaultGP(theta, y, fitAmp=True)

    # Initialize object using the Wang & Li (2017) Rosenbrock function example
    # Use default GP initialization: ExpSquaredKernel
    ap = approx.ApproxPosterior(theta=theta,
                                y=y,
                                gp=gp,
                                lnprior=lh.rosenbrockLnprior,
                                lnlike=lh.rosenbrockLnlike,
                                priorSample=lh.rosenbrockSample,
                                bounds=bounds,
                                algorithm=algorithm)

    # Optimize the GP hyperparameters
    ap.optGP(seed=seed, method="powell", nGPRestarts=3)

    # Find MAP solution
    trueMAP = [1.0, 1.0]
    trueVal = 0.0
    testMAP, testVal = ap.findMAP(nRestarts=5)

    # Compare estimated MAP to true values
    errMsg = "True MAP solution is incorrect."
    # Allow up to 10% error in each parameter
    assert(np.allclose(trueMAP, testMAP, atol=1.0e-1)), errMsg
    # All up to 0.1% error in function value
    errMsg = "True MAP function value is incorrect."
    assert(np.allclose(trueVal, testVal, atol=1.0e-3)), errMsg
# end function


def testMAPNoAmp():
    """
    Test MAP estimation
    """

    # Define algorithm parameters
    m0 = 50                           # Initial size of training set
    bounds = ((-5,5), (-5,5))         # Prior bounds
    algorithm = "bape"                # Use the Kandasamy et al. (2015) formalism
    seed = 57                         # For reproducibility
    np.random.seed(seed)

    # Randomly sample initial conditions from the prior
    theta = np.array(lh.rosenbrockSample(m0))

    # Evaluate forward model log likelihood + lnprior for each theta
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

    # Create the the default GP using an ExpSquaredKernel
    gp = gpUtils.defaultGP(theta, y, fitAmp=True)

    # Initialize object using the Wang & Li (2017) Rosenbrock function example
    # Use default GP initialization: ExpSquaredKernel
    ap = approx.ApproxPosterior(theta=theta,
                                y=y,
                                gp=gp,
                                lnprior=lh.rosenbrockLnprior,
                                lnlike=lh.rosenbrockLnlike,
                                priorSample=lh.rosenbrockSample,
                                bounds=bounds,
                                algorithm=algorithm)

    # Optimize the GP hyperparameters
    ap.optGP(seed=seed, method="powell", nGPRestarts=3)

    # Find MAP solution
    trueMAP = [1.0, 1.0]
    trueVal = 0.0
    testMAP, testVal = ap.findMAP(nRestarts=5)

    # Compare estimated MAP to true values
    errMsg = "True MAP solution is incorrect."
    # Allow up to 10% error in each parameter
    assert(np.allclose(trueMAP, testMAP, atol=1.0e-1)), errMsg
    # All up to 0.1% error in function value
    errMsg = "True MAP function value is incorrect."
    assert(np.allclose(trueVal, testVal, atol=1.0e-3)), errMsg
# end function


if __name__ == "__main__":
    testMAPAmp()
    testMAPNoAmp()
