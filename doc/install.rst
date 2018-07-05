Installation
============

:py:obj:`approxposterior` can be installed several ways.

The preferred installation method is using :py:obj:`conda`:

.. code-block:: bash

  conda install -c conda-forge approxposterior

This will install approxposterior and all dependencies.

Using :py:obj:`pip`:

First install george

.. code-block:: bash

    conda install -c conda-forge george

then install :py:obj:`approxposterior`

.. code-block:: bash

   pip install approxposterior

This step requires installing george (the Python Gaussian Process package) first before
installing approxposterior to ensure george is setup properly.

To upgrade:

.. code-block:: bash

   pip install -U --no-deps approxposterior

From source:

First install george

.. code-block:: bash

    conda install -c conda-forge george

then install :py:obj:`approxposterior`

.. code-block:: bash

  git clone git@github.com:dflemin3/approxposterior.git
  cd approxposterior
  python setup.py install
