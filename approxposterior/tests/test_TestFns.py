#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test implementation of test functions.

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import likelihood as lh
import numpy as np


def testTestFns():
    """
    Test test likelihood and optimization functions from likelihood.py.
    """

    # Check 2D Rosenbrock function, compare to the known global minimum
    test = lh.rosenbrockLnlike([1, 1])
    errMsg = "2D Rosenbrock function is incorrect"
    truth = 0
    assert np.allclose(test, truth), errMsg

    # Check 5D Rosenbrock function, compare to the known global minimum
    test = lh.rosenbrockLnlike([1, 1, 1, 1, 1])
    errMsg = "5D Rosenbrock function is incorrect"
    truth = 0
    assert np.allclose(test, truth), errMsg

    # Check sphere function, compare to the known global minimum
    test = lh.sphereLnlike([0, 0])
    errMsg = "Sphere function is incorrect"
    truth = 0
    assert np.allclose(test, truth), errMsg

    # Check 1D BayesOpt test function, compare to the known global maximum
    test = lh.testBOFn(-0.359)
    errMsg = "1D test BayesOpt function is incorrect"
    truth = 0.5003589
    assert np.allclose(test, truth), errMsg
# end function


if __name__ == "__main__":
    testTestFns()
