"""
Gauss-Newton with direct Hessian inversion
"""

import time
import numpy as np
import numpy.linalg as la

from .utils import residual, einchars, vectorize, unvectorize


def decompose(tensor, rank, nsteps, mu, seed=None):
    """Perform CPD of a tensor using Gauss-Newton with direct
    Hessian inversion
    
    Parameters
    ----------
    tensor: numpy.ndarray
        The tensor to be decomposed
    rank: int
        The target rank
    nsteps: int
        number of Gauss-Newton steps to run
    mu: float
        regularization parameter. H_tilde = H + mu * I
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
    for _ in range(nsteps):
        factors = step(tensor, factors, mu)
    return factors


def step(tensor, factors, mu):
    r = residual(tensor, factors)
    g = apply_jacobian(factors, r)

    t = time.time()
    h = hessian_matrix(factors)
    x = vectorize(factors)

    h_mu = h + mu * np.eye(h.shape[0])

    x -= la.solve(h_mu, g)
    return unvectorize(x, [factor.shape for factor in factors])


def apply_jacobian(factors, r):
    ndim = len(factors)
    result = []
    for i in range(ndim):
        result.append(apply_jacobian_block(factors, r, i))
    return np.block(result)

def apply_jacobian_block(factors, res, p):
    ndim = len(factors)
    k, r = einchars[ndim:ndim+2]
    input_subscripts = [einchars[i] + (k if i == p else r) for i in range(ndim)] + [''.join(einchars[i] for i in range(ndim))]
    output_subscript = k + r
    subscripts = ','.join(input_subscripts) + '->' + output_subscript
    return -np.einsum(subscripts, *[*((np.eye(factors[p].shape[0]) if i == p else factor) for i, factor in enumerate(factors)), res], optimize=True).reshape(-1)

def jacobian_matrix(factors):
    ndim = len(factors)
    jacobians = np.empty(ndim, dtype=object)
    for i in range(ndim):
        jacobians[i] = jacobian(factors, i).reshape(-1, factors[i].size)
    return np.block(jacobians.tolist())


def jacobian(factors, p):
    ndim = len(factors)
    k, r = einchars[ndim:ndim+2]
    input_subscripts = [einchars[i] + (k if i == p else r) for i in range(ndim)]
    output_subscript = ''.join(einchars[i] for i in range(ndim)) + k + r
    subscripts = ','.join(input_subscripts) + '->' + output_subscript
    return -np.einsum(subscripts, *[(np.eye(factors[p].shape[0]) if i == p else factor) for i, factor in enumerate(factors)])

def hessian_matrix(factors):
    ndim = len(factors)
    hessians = np.empty((ndim, ndim), dtype=object)
    for p, q in np.ndindex(ndim, ndim):
        hessians[p, q] = hessian(factors, p, q).reshape(factors[p].size, factors[q].size)
    return np.block(hessians.tolist())

def hessian(factors, p, q):
    g = gamma(factors, p, q)
    if p == q:
        d = factors[p].shape[0]
        return np.einsum('kl,rs->krls', np.eye(d), g)
    else:
        return np.einsum('lr,ks,rs->krls', factors[q], factors[p], g)


def gamma(factors, p, q):
    ndim = len(factors)
    rank = factors[0].shape[1]
    result = np.ones((rank, rank), dtype=factors[0].dtype)
    for m in range(ndim):
        if m != p and m != q:
            result *= np.einsum('ir,is->rs', factors[m], factors[m])
    return result
