import unittest
import numpy as np
import numpy.linalg as la
import cpd


class TestResidual(unittest.TestCase):
    def setUp(self):
        self.rank = 3
        self.dims = [5,5,5]
        self.tensor, _ = cpd.utils.random_low_rank(dims=self.dims, rank=self.rank, seed=0)

    def test_als(self):
        factors = cpd.als.decompose(self.tensor, self.rank, nsweeps=1000, mu=0.0001, seed=1)
        error = la.norm(cpd.utils.residual(self.tensor, factors))
        self.assertLess(error, 0.001)

    def test_gn(self):
        factors = cpd.gn.decompose(self.tensor, self.rank, nsteps=100, mu=0.0001, seed=1)
        error = la.norm(cpd.utils.residual(self.tensor, factors))
        self.assertLess(error, 0.001)

    def test_gn_fast(self):
        factors = cpd.gn_fast.decompose(self.tensor, self.rank, nsteps=100, mu=0.0001, seed=1)
        error = la.norm(cpd.utils.residual(self.tensor, factors))
        self.assertLess(error, 0.001)
