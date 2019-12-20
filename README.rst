Gauss-Newton for Tensor Canonical Polyadic Decomposition (CPD)
==============================================================

This is an implementation of the fast Hessian inversion algorithm
described by `this paper <https://arxiv.org/abs/1205.2584>`_:

::

    Anh Huy Phan, Petr Tichavsky, and Andrzej Cichocki.
    Low Complexity Damped Gauss- Newton Algorithms for CANDECOMP/PARAFAC.


Installation
------------

.. code-block:: console

    pip install -e path/to/the/project/directory


Testing
-------
In the project directory, run

.. code-block:: console

    python -m unittest


Usage
-----
There are three modules in this package:
`als <cpd/als.py>`_,
`gn <cpd/gn.py>`_ and
`gn_fast <cpd/gn_fast.py>`_.
Each of them implements a function called `decompose`
that decomposes a tensor using the corresponding method
(see the documentation in the source code).
g