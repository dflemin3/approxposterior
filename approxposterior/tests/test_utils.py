"""

Test the utility functions.

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import numpy as np
import scipy.stats as ss
from approxposterior import utility as ut, gp_utils


def test_utils_gp():
    """
    Test the utility functions!  This probes the gp_utils.setup_gp function
    (which is rather straight-forward) and makes sure the utility functions
    produce the right result (which is also straight-forward).  If something is
    broke, you'll get an assert error

    Parameters
    ----------

    Returns
    -------
    """

    # Define 20 input points
    theta = [[-3.19134011, -2.91421701],
         [-1.18523861,  1.19142021],
         [-0.18079129, -2.28992237],
         [-4.3620168 ,  3.21190854],
         [-1.15409246,  4.09398417],
         [-3.12326862,  2.79019658],
         [ 4.17301261, -0.04203107],
         [-2.2891247 ,  4.09017865],
         [ 0.35073782,  4.60727199],
         [ 0.45842724,  0.17847542],
         [ 3.64210638,  1.68852975],
         [-4.24319817, -1.01012339],
         [-2.25954639, -3.01525571],
         [-4.42631479,  0.25335136],
         [-1.21944971,  4.14651088],
         [4.36308741, -2.88213344],
         [-3.05242599, -4.18389666],
         [-2.64969466, -2.55430067],
         [ 2.16145337, -4.80792732],
         [-2.47627867, -1.40710833]]

    y = [-1.71756034e+02,  -9.32795815e-02,  -5.40844997e+00,
        -2.50410657e+02,  -7.67534763e+00,  -4.86758102e+01,
        -3.04814897e+02,  -1.43048383e+00,  -2.01127580e+01,
        -3.93664011e-03,  -1.34083055e+02,  -3.61839586e+02,
        -6.60537302e+01,  -3.74287939e+02,  -7.12195137e+00,
        -4.80540985e+02,  -1.82446653e+02,  -9.18173221e+01,
        -8.98802494e+01,  -5.69583369e+01]

    # Set up a gp
    gp = gp_utils.setup_gp(theta, y, which_kernel="ExpSquaredKernel")
    gp = gp_utils.optimize_gp(gp, theta, y, which_kernel="ExpSquaredKernel")

    # Compute the AGP utility function at some point
    theta_test = np.array([-2.3573, 4.673])
    test_util = ut.AGP_utility(theta_test, y, gp)

    print(test_util)

    err_msg = "ERROR: AGP util fn bug.  Did you change gp_utils.setup_gp?"
    assert np.allclose(test_util,11.35206957), err_msg

    # Now do the same using the BAPE utility function
    test_util = ut.BAPE_utility(theta_test, y, gp)

    err_msg = "ERROR: BAPE util fn bug.  Did you change gp_utils.setup_gp?"
    assert np.allclose(test_util,21.51239966), err_msg

    return None
# end function


def test_kl_approximation():
    """
    Test the accuracy of the Monte Carlo approximation for computing the
    KL divergence.

    Parameters
    ----------

    Returns
    -------
    """

    # Set RNG Seed
    np.random.seed(42)

    num = 1000
    x = np.linspace(-5, 5, num)

    # Make two different normal pdfs
    p_diff = ss.norm.pdf(x, loc=1.2, scale=1)
    q_diff = ss.norm.pdf(x, loc=-1.2, scale=1)

    # Estimate KL divergence: Should be rather non-zero
    kl_diff = ss.entropy(p_diff, q_diff)

    # Now numerically estimate the KL-divergence
    p_kwargs = {"loc": 1.2, "scale": 1}
    q_kwargs = {"loc" : -1.2, "scale" : 1}

    # Wrap the functions
    p_pdf = ut.function_wrapper(ss.norm.pdf, **p_kwargs)
    q_pdf = ut.function_wrapper(ss.norm.pdf, **q_kwargs)

    x = ss.norm.rvs(loc=1.2, scale=1, size=10000)
    numerical = ut.kl_numerical(x, p_pdf, q_pdf)

    # Answer better be close (small percent difference)
    err_msg = "ERROR: Numerical approximation incorrect by > 0.5%!"
    assert(100*np.fabs((kl_diff - numerical)/kl_diff) < 0.5), err_msg

    return None
# end function
