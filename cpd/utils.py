from string import ascii_letters as einchars

import numpy as np
import numpy.linalg as la


def residual(tensor, factors):
    """Compute the residual of a factorization

    Parameters
    ----------
    tensor: numpy.ndarray
        The true tensor

    factors: numpy.ndarray
        An object array contains the factor tensors

    Returns
    -------
    numpy.ndarray
        The residual tensor of the same shape as the true tensor
    """
    return tensor - reconstruct(factors)


def reconstruct(factors):
    """Reconstruct the tensor from its CPD factors

    Parameters
    ----------
    factors: numpy.ndarray
        An object array contains the factor tensors 

    Returns
    -------
    numpy.ndarray
        The reconstructed trensor
    """
    ndim = len(factors)
    input_subscripts = [einchars[i] + einchars[ndim] for i in range(ndim)]
    output_subscript = ''.join(einchars[i] for i in range(ndim))
    subscripts = ','.join(input_subscripts) + '->' + output_subscript
    return np.einsum(subscripts, *factors)


def random_low_rank(dims, rank, seed=None):
    """Create a random low rank tensor

    Parameters
    ----------
    dims: List[int]
        The size of each dimensions
    
    rank: int
        The tensor rank

    seed: int or None
        The seed for random number generation

    Returns
    -------
    numpy.ndarray
        The random low rank tensor
    """
    rstate = np.random.RandomState(seed)
    factors = np.empty(len(dims), dtype=object)
    for i, d in enumerate(dims):
        factors[i] = rstate.rand(d, rank)
    tensor = reconstruct(factors)
    return tensor, factors


def vectorize(tensors):
    return np.block([tensor.reshape(-1) for tensor in tensors])


def unvectorize(vector, shapes):
    tensors = np.empty(len(shapes), dtype=object)
    start = 0
    for i, shape in enumerate(shapes):
        size = np.prod(shape)
        tensors[i] = vector[start:start+size].reshape(*shape)
        start += size
    return tensors

