"""

Test the GMM fitting procedure.

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

import numpy as np
import sys
from approxposterior import gmmUtils

def testGMMFit():
    """
    Test the accuracy our GMM fitting scheme.

    Parameters
    ----------

    Returns
    -------
    """

    # Set RNG Seed
    np.random.seed(57)

    # Spherical gaussian centered on (5, 10)
    shiftG = np.random.randn(500, 2) + np.array([5, 10])

    # Save mean
    muShiftG = np.mean(shiftG, axis=0)

    # Zero centered Gaussian data
    c = np.array([[0., -0.7], [3.5, .7]])
    stretchG = np.dot(np.random.randn(300, 2), c)

    # Save mean
    muStetchG = np.mean(stretchG, axis=0)

    # Combine dataset, randomize points
    data = np.vstack([shiftG, stretchG])
    np.random.shuffle(data)

    # Fit!
    gmm = gmmUtils.fitGMM(data, maxComp=10, covType="full")

    # Did it infer 2 components for data generated from two disjoint dists?
    errMsg = "ERROR: fitGMM did not infer 2 components! n_components = %d" % gmm.n_components
    assert(2 == gmm.n_components), errMsg

    # Behavior is different for diff versions of python - so account for that!
    if sys.version_info[1] <= 5:
        stretchInd = 0
        shiftInd = 1
    else:
        stretchInd = 1
        shiftInd = 0

    # Ensure that the true and inferred Gaussian means are the same
    errMsg = "ERROR: fitGMM inferred incorrect means!"
    assert(np.allclose(muStetchG, gmm.means_[stretchInd]))
    assert(np.allclose(muShiftG, gmm.means_[shiftInd]))
# end function

if __name__ == "__main__":
    testGMMFit()
