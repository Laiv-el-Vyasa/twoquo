import numpy as np


def matrix_to_qubo(qubo_matrix):
    qubo = {}
    for pos, val in np.ndenumerate(qubo_matrix):
        if val:
            qubo[pos] = val
    return qubo


def qubo_to_matrix(qubo, size):
    Q = np.zeros((size, size))
    for (x, y), v in qubo.items():
        Q[x][y] = v
    return Q


def to_compact_matrix(qubo_matrix):
    """Returns a compact view of the QUBO matrix.

    The QUBO matrix is expected to be a numpy array.
    Returns the upper triangular part of the QUBO matrix, flattened, and the
    size of the QUBO matrix. This includes the diagonal.
    """
    if qubo_matrix is None:
        return None, None
    size = qubo_matrix.shape[0]
    return qubo_matrix[np.triu_indices(size)].flatten(), size


def to_detailed_matrix(qubo_matrix, size):
    """Returns a fully quadratic view of the QUBO matrix.

    The complement of this function is `to_compact_matrix`.
    The QUBO matrix is expected to be a flat numpy array.
    """
    if qubo_matrix is None:
        return None

    Q = np.zeros((size, size))
    Q[np.triu_indices(size)] = qubo_matrix
    diag = np.diag(Q)
    Q = np.triu(Q) + np.triu(Q, 1).T
    np.fill_diagonal(Q, diag)
    return Q
