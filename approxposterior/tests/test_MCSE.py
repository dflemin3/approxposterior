#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test estimating the Monte Carlo Standard Error (MCSE) of an MCMC chain

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import mcmcUtils
import numpy as np


def testMCSE():
    """
    Test MCSE estimation based on the nice example of the MCSE of an Ar(1) chain from
    https://stats.stackexchange.com/questions/201790/mcmc-convergence-analytic-derivations-monte-carlo-error
    """

    np.random.seed(57)

    # Simulate simple MCMC chain based on Ar(1), an autoregressive process
    num  = int(1.0e5)
    samples = np.zeros(num)

    for ii in range(1,num):
        samples[ii] = 0.4 * samples[ii-1] + np.random.randn()
    mcse = mcmcUtils.batchMeansMCSE(samples)

    # Compare estimated MCSE to the known value
    errMsg = "MCSE is incorrect"
    trueMCSE = 0.00494
    assert np.allclose(trueMCSE, mcse, atol=2.5e-3), errMsg
# end function


if __name__ == "__main__":
    testMCSE()
