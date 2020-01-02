"""

Test the GP optimizer and utility functions.

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

import numpy as np
import george
from approxposterior import utility as ut, gpUtils as gpu, likelihood as lh


def testGPOptAmp():
    """
    Test optimizing the GP hyperparameters.

    Parameters
    ----------

    Returns
    -------
    """

    # For reproducibility
    m0 = 50
    seed = 57
    np.random.seed(seed)

    # Randomly sample initial conditions from the prior
    theta = np.array(lh.rosenbrockSample(m0))

    # Evaluate forward model log likelihood + lnprior for each theta
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

    # Set up a gp
    gp = gpu.defaultGP(theta, y, fitAmp=True)

    # Optimize gp using default opt parameters
    gp = gpu.optimizeGP(gp, theta, y, seed=seed, nGPRestarts=5)

    # Extract GP hyperparameters, compare to truth
    # Ignore mean fit - just focus on scale lengths and amplitude
    hypeTest = gp.get_parameter_vector()[1:]

    errMsg = "ERROR: GP hyperparameters are not close to the true value!"
    hypeTrue = [19.99668368, 4.18856645, 10.78000803]
    assert np.allclose(hypeTest, hypeTrue, rtol=1.0e-2), errMsg
# end function


def testGPOptNoAmp():
    """
    Test optimizing the GP hyperparameters.

    Parameters
    ----------

    Returns
    -------
    """

    # For reproducibility
    m0 = 50
    seed = 57
    np.random.seed(seed)

    # Randomly sample initial conditions from the prior
    theta = np.array(lh.rosenbrockSample(m0))

    # Evaluate forward model log likelihood + lnprior for each theta
    y = np.zeros(len(theta))
    for ii in range(len(theta)):
        y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

    # Set up a gp
    gp = gpu.defaultGP(theta, y, fitAmp=False)

    # Optimize gp using default opt parameters
    gp = gpu.optimizeGP(gp, theta, y, seed=seed, nGPRestarts=5)

    # Extract GP hyperparameters, compare to truth
    # Ignore mean fit - just focus on scale lengths and amplitude
    hypeTest = gp.get_parameter_vector()[1:]
    print(hypeTest)

    errMsg = "ERROR: GP hyperparameters are not close to the true value!"
    hypeTrue = [-1.54256578, 3.24723589]
    assert np.allclose(hypeTest, hypeTrue, rtol=1.0e-2), errMsg
# end function

if __name__ == "__main__":
    testGPOptAmp()
    testGPOptNoAmp()
