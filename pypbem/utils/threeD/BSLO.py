from .green_evaluation import green_eval
import numpy as np
from numba import jit


@jit(nopython=True)
def assemble_BSLO(grid, k, v, d) -> np.ndarray:
    """
    Assembles the 3D Boundary Single Layer Operator for a periodic Green function.

    Parameters
    ----------
    grid : np.ndarray
        _size(N,3), grid of the sphere_
    k : float
        _wavenumber_
    v : np.ndarray
        _size (1,3), vector of incident wave_
    d : float
        _period of the problem_

    Returns
    -------
    np.ndarray
        _Matrix of size (N,N) where N is the number of points in the grid_
    """
    beta = d * k * v
    M = 50

    N = np.shape(grid)[0]
    BSLO = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(i + 1):
            temp = green_eval(grid[i, :] - grid[j, :], beta, k, d, M)
            BSLO[i, j] = temp
            BSLO[j, i] = temp

    return BSLO
