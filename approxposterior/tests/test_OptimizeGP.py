"""

Test the GP optimizer and utility functions.

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

import numpy as np
import george
from approxposterior import utility as ut, gpUtils as gpu


def testGPOpt():
    """
    Test optimizing the GP hyperparameters.

    Parameters
    ----------

    Returns
    -------
    """

    # For reproducibility
    seed = 91
    np.random.seed(seed)

    # Define 20 input points
    theta = np.array([[-3.19134011, -2.91421701],
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
         [-2.47627867, -1.40710833]]).squeeze()

    y = np.array([-1.71756034e+02,  -9.32795815e-02,  -5.40844997e+00,
        -2.50410657e+02,  -7.67534763e+00,  -4.86758102e+01,
        -3.04814897e+02,  -1.43048383e+00,  -2.01127580e+01,
        -3.93664011e-03,  -1.34083055e+02,  -3.61839586e+02,
        -6.60537302e+01,  -3.74287939e+02,  -7.12195137e+00,
        -4.80540985e+02,  -1.82446653e+02,  -9.18173221e+01,
        -8.98802494e+01,  -5.69583369e+01]).squeeze()

    # Set up a gp
    gp = gpu.defaultGP(theta, y)

    # Optimize gp using default opt parameters
    method = "nelder-mead"
    options = {"adaptive" : True}
    p0 = gp.get_parameter_vector()

    # Try serial computation
    gp = gpu.optimizeGP(gp, theta, y, seed=seed, nGPRestarts=5,
                        method=method, options=options, p0=p0,
                        nCores=1)

    # Extract GP hyperparameters, compare to truth
    hypeSin = gp.get_parameter_vector()

    # Now with multiprocessing using as many cores as we can
    gp = gpu.optimizeGP(gp, theta, y, seed=seed, nGPRestarts=5,
                        method=method, options=options, p0=p0,
                        nCores=-1)

    # Extract GP hyperparameters, compare to truth
    hypeMult = gp.get_parameter_vector()

    errMsg = "ERROR: GP hyperparameters are not close to the true value!"
    errMult = "ERROR: Single and multi GP hyperparameters are not close!"
    hypeTrue = [-135.91343934, -0.68256741, 2.1394469]
    assert np.allclose(hypeSin, hypeMult), errMult
    assert np.allclose(hypeSin, hypeTrue), errMsg
    assert np.allclose(hypeMult, hypeTrue), errMsg
# end function

if __name__ == "__main__":
    testGPOpt()
