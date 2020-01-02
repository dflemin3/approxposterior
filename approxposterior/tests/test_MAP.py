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
    m0 = 20                           # Initial size of training set
    bounds = [(-5,5), (-5,5)]         # Prior bounds
    algorithm = "jones"
    seed = 57                         # For reproducibility
    np.random.seed(seed)

    # Randomly sample initial conditions from the prior
    theta = np.array(lh.sphereSample(m0))

    # Evaluate forward model log likelihood + lnprior for each theta
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lh.sphereLnlike(theta[ii]) + lh.sphereLnprior(theta[ii])

    # Create the the default GP using an ExpSquaredKernel
    gp = gpUtils.defaultGP(theta, y, fitAmp=True)

    # Initialize object using the Wang & Li (2017) Rosenbrock function example
    # Use default GP initialization: ExpSquaredKernel
    ap = approx.ApproxPosterior(theta=theta,
                                y=y,
                                gp=gp,
                                lnprior=lh.sphereLnprior,
                                lnlike=lh.sphereLnlike,
                                priorSample=lh.sphereSample,
                                bounds=bounds,
                                algorithm=algorithm)

    # Optimize the GP hyperparameters
    ap.optGP(seed=seed, method="powell", nGPRestarts=3)

    # Find some points to add to GP training set
    ap.findNextPoint(numNewPoints=5, nGPRestarts=3, cache=False)

    # Find MAP solution
    trueMAP = [0.0, 0.0]
    trueVal = 0.0
    testMAP, testVal = ap.findMAP(nRestarts=15)

    # Compare estimated MAP to true values, given some tolerance
    errMsg = "True MAP solution is incorrect."
    assert(np.allclose(trueMAP, testMAP, atol=1.0e-3)), errMsg
    errMsg = "True MAP function value is incorrect."
    assert(np.allclose(trueVal, testVal, atol=1.0e-3)), errMsg
# end function


if __name__ == "__main__":
    testMAPAmp()
