#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Test loading approxposterior

@author: David P. Fleming [University of Washington, Seattle], 2018
@email: dflemin3 (at) uw (dot) edu

"""


def test_import():
    """
    Test importing approxposterior
    """

    import approxposterior
    import emcee
    version = emcee.__version__
    errMsg = "approxposterior is only compatible with emcee versions >= 3"
    assert int(version.split(".")[0]) > 2, errMsg
# end function

if __name__ == "__main__":
    test_import()
