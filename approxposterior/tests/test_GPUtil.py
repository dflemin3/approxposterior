"""

Test optimizing GP utility functions.

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

import numpy as np
import george
from approxposterior import utility as ut, gpUtils, likelihood as lh


def testUtilsGPAmp():
    """
    Test the utility functions!  This probes the gp_utils.setup_gp function
    (which is rather straight-forward) and makes sure the utility functions
    produce the right result (which is also straight-forward). Based on the
    Wang+2017 Rosenbrock function example.

    Parameters
    ----------

    Returns
    -------
    """

    # For reproducibility
    m0 = 20
    seed = 57
    np.random.seed(seed)

    # Randomly sample initial conditions from the prior
    theta = np.array(lh.rosenbrockSample(m0))

    # Evaluate forward model log likelihood + lnprior for each theta
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

    # Set up a gp
    gp = gpUtils.defaultGP(theta, y, fitAmp=True)

    # Compute the AGP utility function at some point
    thetaTest = np.array([-2.3573, 4.673])
    testUtil = ut.AGPUtility(thetaTest, y, gp, lh.rosenbrockLnprior)

    errMsg = "ERROR: AGP util fn bug.  Did you change gp_utils.setup_gp?"
    assert np.allclose(testUtil, 38.74563838, rtol=1.0e-4), errMsg

    # Now do the same using the BAPE utility function
    testUtil = ut.BAPEUtility(thetaTest, y, gp, lh.rosenbrockLnprior)

    errMsg = "ERROR: BAPE util fn bug.  Did you change gp_utils.setup_gp?"
    assert np.allclose(testUtil, -114623.57332731, rtol=1.0e-4), errMsg

    # Now do the same using the naive utility function
    testUtil = ut.NaiveUtility(thetaTest, y, gp, lh.rosenbrockLnprior)

    errMsg = "ERROR: naive util fn bug.  Did you change gp_utils.setup_gp?"
    assert np.allclose(testUtil, 2226233.22000008, rtol=1.0e-4), errMsg
# end function


def testUtilsGPNoAmp():
    """
    Test the utility functions!  This probes the gp_utils.setup_gp function
    (which is rather straight-forward) and makes sure the utility functions
    produce the right result (which is also straight-forward). Based on the
    Wang+2017 Rosenbrock function example.

    Parameters
    ----------

    Returns
    -------
    """

    # For reproducibility
    m0 = 20
    seed = 57
    np.random.seed(seed)

    # Randomly sample initial conditions from the prior
    theta = np.array(lh.rosenbrockSample(m0))

    # Evaluate forward model log likelihood + lnprior for each theta
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

    # Set up a gp
    gp = gpUtils.defaultGP(theta, y, fitAmp=False)

    # Compute the AGP utility function at some point
    thetaTest = np.array([-2.3573, 4.673])
    testUtil = ut.AGPUtility(thetaTest, y, gp, lh.rosenbrockLnprior)

    errMsg = "ERROR: AGP util fn bug.  Did you change gp_utils.setup_gp?"
    assert np.allclose(testUtil, 38.53018886, rtol=1.0e-4), errMsg

    # Now do the same using the BAPE utility function
    testUtil = ut.BAPEUtility(thetaTest, y, gp, lh.rosenbrockLnprior)

    errMsg = "ERROR: BAPE util fn bug.  Did you change gp_utils.setup_gp?"
    assert np.allclose(testUtil, 76.26762099, rtol=1.0e-4), errMsg

    # Now do the same using the naive utility function
    testUtil = ut.NaiveUtility(thetaTest, y, gp, lh.rosenbrockLnprior)

    errMsg = "ERROR: naive util fn bug.  Did you change gp_utils.setup_gp?"
    assert np.allclose(testUtil, 38.28476014, rtol=1.0e-4), errMsg
# end function

if __name__ == "__main__":
    testUtilsGPAmp()
    testUtilsGPNoAmp()
