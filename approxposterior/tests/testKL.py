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
    np.random.seed(42)

    num = 1000
    x = np.linspace(-5, 5, num)

    # Make two different normal pdfs
    pDiff = ss.norm.pdf(x, loc=1.2, scale=1)
    qDiff = ss.norm.pdf(x, loc=-1.2, scale=1)

    # Estimate KL divergence: Should be rather non-zero
    KLDiff = ss.entropy(pDiff, qDiff)

    # Now numerically estimate the KL-divergence
    p_kwargs = {"loc": 1.2, "scale": 1}
    q_kwargs = {"loc" : -1.2, "scale" : 1}

    # Wrap the functions
    p_pdf = ut.functionWrapper(ss.norm.pdf, **p_kwargs)
    q_pdf = ut.functionWrapper(ss.norm.pdf, **q_kwargs)

    x = ss.norm.rvs(loc=1.2, scale=1, size=10000)
    numerical = ut.klNumerical(x, p_pdf, q_pdf)

    # Answer better be close (small percent difference)
    err_msg = "ERROR: Numerical KL divergence approximation incorrect by >0.5%!"
    assert(100*np.fabs((KLDiff - numerical)/KLDiff) < 0.5), err_msg

    return None
# end function
