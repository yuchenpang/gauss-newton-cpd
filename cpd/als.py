"""
Alternative least square
"""

import numpy as np
import numpy.linalg as la

from .utils import residual, einchars, vectorize, unvectorize


def decompose(tensor, rank, nsweeps, mu, seed=None):
    """Perform CPD of a tensor using ALS method
    
    Parameters
    ----------
    tensor: numpy.ndarray
        The tensor to be decomposed
    rank: int
        The target rank
    nsweeps: int
        number of ALS sweeps
    mu: float
        regularization parameter. Modify each local linear system
            Ax = b into (A + mu * I) x = b.
    seed: int or None
        The seed for randomized initialization

    Returns
    -------
    numpy.ndarray
        An object array of factor tensors
    """
    rstate = np.random.RandomState(seed)
    ndim = tensor.ndim
    factors = np.empty(ndim, dtype=object)
    for i in range(ndim):
        factors[i] = rstate.rand(tensor.shape[i], rank)
    for _ in range(nsweeps):
        for d in range(ndim):
            factors[d] = step(tensor, factors, d, mu)
    return factors


def step(tensor, factors, d, mu):
    rank = factors[0].shape[1]
    return rhs(tensor, factors, d) @ la.inv(gamma(factors, d, d) + mu * np.eye(rank))


def rhs(tensor, factors, d):
    ndim = tensor.ndim
    r = einchars[ndim]
    i = einchars[:ndim]
    input_subscripts = [(i[p] + r) if p != d else i for p in range(ndim)]
    output_subscripts = i[d] + r
    subscripts = ','.join(input_subscripts) + '->' + output_subscripts
    return np.einsum(subscripts, *[factor if p != d else tensor for p,factor in enumerate(factors)])


def gamma(factors, p, q):
    ndim = len(factors)
    rank = factors[0].shape[1]
    result = np.ones((rank, rank), dtype=factors[0].dtype)
    for m in range(ndim):
        if m != p and m != q:
            result *= np.einsum('ir,is->rs', factors[m], factors[m])
    return result
