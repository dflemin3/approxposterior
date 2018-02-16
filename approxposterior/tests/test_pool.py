#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test the pool functionality.

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""

import numpy as np
from approxposterior.pool import Pool

def _test_function(x):
    """
    Wastes a random amount of time, then
    returns the average of :py:obj:`x`.
    """

    for i in range(np.random.randint(99999)):
        j = i ** 2

    return np.sum(x) / float(len(x))

def test_pool():
    """
    Test the MultiPool functionality.

    Parameters
    ----------

    Returns
    -------
    """

    # Instantiate the pool
    with Pool(pool="MultiPool") as pool:

        # The iterable we'll apply ``_test_function`` to
        walkers = np.array([[i, i] for i in range(100)], dtype = 'float64')

        # Use the pool to map ``walkers`` onto the function
        res = pool.map(_test_function, walkers)

        # Check if the parallelization worked
        assert np.allclose(res, [_test_function(w) for w in walkers])
# end function
