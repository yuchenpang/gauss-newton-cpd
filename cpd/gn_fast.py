"""
Gauss-Newton with fast Hessian inversion

Reference:
----------

Anh Huy Phan, Petr Tichavsky ́, and Andrzej Cichocki.
Low Complexity Damped Gauss- Newton Algorithms for CANDECOMP/PARAFAC.
https://arxiv.org/abs/1205.2584
"""

import time

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

import matplotlib.pyplot as plt

from .utils import residual, einchars, vectorize, unvectorize


def decompose(tensor, rank, nsteps, mu, seed=None):
    """Perform CPD of a tensor using Gauss-Newton method with fast
    Hessian invesion

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
        factors = step(tensor, factors, rank, mu)
    return factors


def step(tensor, factors, rank, mu):
    ndim = tensor.ndim
    r = residual(tensor, factors)
    g = apply_jacobian(factors, r)

    c_blocks = compute_c_blocks(factors)
    gamma_blocks = compute_gamma_blocks(c_blocks)
    k_blocks = compute_k_blocks(gamma_blocks, rank, ndim)
    k_matrix = compute_k_matrix(k_blocks, rank, ndim)
    gamma_mu_diag_blocks_inv = compute_gamma_mu_diag_blocks_inv(gamma_blocks, rank, ndim, mu)
    psi_mu_blocks = compute_psi_mu_blocks(gamma_mu_diag_blocks_inv, c_blocks, ndim)
    psi_mu = sla.block_diag(*psi_mu_blocks)

    z_matrix = compute_z_matrix(factors, ndim, rank)
    b_mu = la.inv(la.inv(k_matrix) + psi_mu)
    g_mu_inv_blocks = compute_g_mu_inv_blocks(factors, gamma_mu_diag_blocks_inv, ndim, rank)
    l_mu_blocks = compute_l_mu_blocks(gamma_mu_diag_blocks_inv, factors, ndim, rank)
    l_mu = sla.block_diag(*l_mu_blocks)

    def apply_g_mu_inv(g):
        g_parts = []
        start = 0
        for i in range(ndim):
            size = factors[i].size
            g_parts.append(g[start:start+size])
            start += size
        result = []
        for g_mu_inv_block, g_part in zip(g_mu_inv_blocks, g_parts):
            result.append(g_mu_inv_block @ g_part)
        return np.block(result)
    
    def apply_l_mu_T(g):
        g_parts = []
        start = 0
        for i in range(ndim):
            size = factors[i].size
            g_parts.append(g[start:start+size])
            start += size
        result = []
        for l_mu_block, g_part in zip(l_mu_blocks, g_parts):
            result.append(l_mu_block.T @ g_part)
        return np.block(result)

    def apply_l_mu(x):
        x_parts = x.reshape(ndim,-1)
        result = []
        for l_mu_block, x_part in zip(l_mu_blocks, x_parts):
            result.append(l_mu_block @ x_part)
        return np.block(result)

    def apply_h_mu_inv(g):
        return apply_g_mu_inv(g) - apply_l_mu(b_mu @ (apply_l_mu_T(g)))

    x = vectorize(factors) - apply_h_mu_inv(g)

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

def compute_hessian_matrix(factors, gamma_blocks):
    ndim = len(factors)
    hessians = np.empty((ndim, ndim), dtype=object)
    for p, q in np.ndindex(ndim, ndim):
        hessians[p, q] = compute_hessian_block(factors, gamma_blocks, p, q).reshape(factors[p].size, factors[q].size)
    return np.block(hessians.tolist())

def compute_hessian_diag_matrix(factors, gamma_blocks):
    ndim = len(factors)
    hessians = np.empty(ndim, dtype=object)
    for p, in np.ndindex(ndim):
        hessians[p] = compute_hessian_block(factors, gamma_blocks, p, p).reshape(factors[p].size, factors[p].size)
    return sla.block_diag(*hessians.tolist())

def compute_hessian_block(factors, gamma_blocks, p, q):
    if p == q:
        d = factors[p].shape[0]
        return np.einsum('kl,rs->krls', np.eye(d), gamma_blocks[p])
    else:
        return np.einsum('lr,ks,rs->krls', factors[q], factors[p], gamma_blocks[p, q])

def gamma_mu_inv(factors, rank, p, q, mu):
    return la.inv(gamma(factors, p, q) + mu * np.eye(rank))

def compute_c_blocks(factors):
    ndim = len(factors)
    c_blocks = np.empty_like(factors)
    for d in range(ndim):
        c_blocks[d] = factors[d].T @ factors[d]
    return c_blocks

def compute_gamma_blocks(c_blocks):
    ndim = len(c_blocks)
    gamma_blocks = {}
    for i in range(ndim):
        for j in range(i, ndim):
            gamma_ij = np.prod([c for idx, c in enumerate(c_blocks) if idx != i and idx != j], axis=0)
            if i == j:
                gamma_blocks[i] = gamma_ij
            else:
                gamma_blocks[i, j] = gamma_blocks[j, i] = gamma_ij
    gamma_blocks[None] = np.prod(c_blocks, axis=0)
    return gamma_blocks

def compute_k_blocks(gamma_blocks, rank, ndim):
    permutation = np.arange(rank**2).reshape(rank,rank).T.reshape(-1)
    k_blocks = {}
    for i in range(ndim):
        for j in range(i+1, ndim):
            k_ij = np.diag(gamma_blocks[i, j].reshape(-1))[permutation]
            k_blocks[i, j] = k_blocks[j, i] = k_ij.T
    return k_blocks

def compute_k_matrix(k_blocks, rank, ndim):
    k = np.empty((ndim, ndim), dtype=object)
    zeros = np.zeros((rank**2, rank**2))
    for i in range(ndim):
        for j in range(ndim):
            k[i,j] = zeros if i == j else k_blocks[i, j]
    return np.block(k.tolist())

def compute_z_matrix(factors, ndim, rank):
    return sla.block_diag(*(np.einsum('rs,kt->krst', np.eye(rank), factors[i]).reshape(-1,rank**2) for i in range(ndim)))

def compute_g_mu_blocks(factors, gamma_blocks, ndim, rank, mu):
    g_mu_blocks = []
    for i in range(ndim):
        d = factors[i].shape[0]
        s = factors[i].size
        g_mu_blocks.append(np.einsum('kl,rs->krls', np.eye(d), gamma_blocks[i]).reshape(s, s) + mu*np.eye(s,s))
    return g_mu_blocks

def compute_g_matrix(factors, gamma_blocks, ndim, rank):
    h_diagonals = []
    for i in range(ndim):
        d = factors[i].shape[0]
        s = factors[i].size
        h_diagonals.append(np.einsum('kl,rs->krls', np.eye(d), gamma_blocks[i]).reshape(s, s))
    return sla.block_diag(*h_diagonals)

def compute_g_mu_inv_blocks(factors, gamma_mu_diag_blocks_inv, ndim, rank):
    g_mu_inv_blocks = []
    for i in range(ndim):
        d = factors[i].shape[0]
        s = factors[i].size
        g_mu_inv_blocks.append(np.einsum('kl,rs->krls', np.eye(d), gamma_mu_diag_blocks_inv[i]).reshape(s, s))
    return g_mu_inv_blocks

def compute_gamma_mu_diag_blocks_inv(gamma_blocks, rank, ndim, mu):
    return [la.inv(gamma_blocks[i]+mu*np.eye(rank)) for i in range(ndim)]

def compute_l_mu_matrix(gamma_mu_diag_blocks_inv, factors, ndim, rank):
    diagonals = []
    for i in range(ndim):
        d = factors[i].shape[0]
        s = factors[i].size
        diagonals.append(np.einsum('kl,rs->krsl', factors[i], gamma_mu_diag_blocks_inv[i]).reshape(s, -1))
    return sla.block_diag(*diagonals)

def compute_l_mu_blocks(gamma_mu_diag_blocks_inv, factors, ndim, rank):
    diagonals = []
    for i in range(ndim):
        d = factors[i].shape[0]
        s = factors[i].size
        diagonals.append(np.einsum('kl,rs->krsl', factors[i], gamma_mu_diag_blocks_inv[i]).reshape(s, -1))
    return diagonals

def compute_damped_als_updates(tensor, factors, gamma_mu_blocks_inv):
    ndim = len(factors)
    damped_als_updates = np.empty_like(factors)
    for i in range(ndim):
        rhs = compute_als_rhs(tensor, factors, i)
        damped_als_updates[i] = rhs @ gamma_mu_blocks_inv[i]
    return damped_als_updates

def compute_als_rhs(tensor, factors, d):
    ndim = tensor.ndim
    r = einchars[ndim]
    i = einchars[:ndim]
    input_subscripts = [(i[p] + r) if p != d else i for p in range(ndim)]
    output_subscripts = i[d] + r
    subscripts = ','.join(input_subscripts) + '->' + output_subscripts
    return np.einsum(subscripts, *[factor if p != d else tensor for p,factor in enumerate(factors)])

def compute_w_mu(factors, damped_als_updates, gamma_blocks, gamma_mu_inv, c_blocks):
    ndim = len(factors)
    w_mu = [(factors[i].T @ damped_als_updates[i] - gamma_blocks[None] @ gamma_mu_inv[i]) for i in range(ndim)]
    return np.block(w_mu).reshape(-1)

def compute_psi_mu_blocks(gamma_mu_inv, c_blocks, ndim):
    return [np.kron(gamma_mu_inv[i], c_blocks[i]) for i in range(ndim)]
