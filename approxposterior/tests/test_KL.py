"""

Test the numerical KL divergence computation.

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import numpy as np
import scipy.stats as ss
from approxposterior import utility as ut

def testKLApproximation():
    """
    Test the accuracy of the Monte Carlo approximation for computing the
    KL divergence.

    Parameters
    ----------

    Returns
    -------
    """

    # Set RNG Seed
    np.random.seed(57)

    num = 1000
    x = np.linspace(-5, 5, num)

    # Make two different normal pdfs
    pDiff = ss.norm.pdf(x, loc=1.2, scale=1)
    qDiff = ss.norm.pdf(x, loc=-1.2, scale=1)

    # Estimate KL divergence: Should be rather non-zero
    KLDiff = ss.entropy(pDiff, qDiff)

    # Now numerically estimate the KL-divergence
    pKwargs = {"loc": 1.2, "scale": 1}
    qKwargs = {"loc" : -1.2, "scale" : 1}

    # Package as lambda functions
    pPdf = lambda x : ss.norm.pdf(x, **pKwargs)
    qPdf = lambda x : ss.norm.pdf(x, **qKwargs)

    x = ss.norm.rvs(loc=1.2, scale=1, size=10000)
    numerical = ut.klNumerical(x, pPdf, qPdf)

    # Answer better be close (small percent difference)
    errMsg = "ERROR: Numerical KL divergence approximation incorrect by >0.5%!"
    assert(100*np.fabs((KLDiff - numerical)/KLDiff) < 0.5), errMsg
# end function

if __name__ == "__main__":
    testKLApproximation()
