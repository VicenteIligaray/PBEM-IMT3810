import numpy as np
from .green_evaluation import green_eval
from numba import jit


@jit(nopython=True)
def potential_eval(point, grid, alpha, k, v, d) -> np.complex64:
    """
    Evaluates the 3D single layer potential operator by approximating the
    integral, evaluating at the point the Green function

    Parameters
    ----------
    point : np.ndarray
        _size (1,2), point to evaluate_
    grid : np.ndarray
        _size(N,3), grid of the sphere_
    alpha : np.ndarray
        _size (N,) coefficients of scattered field_
    k : float
        _wavenumber_
    v : np.ndarray
        _size (1,3), vector of incident wave_
    d : float
        _period of the problem_

    Returns
    -------
    np.complex64
        _evaluation of the operator_
    """
    beta = d * k * v
    M = 50

    result = 0
    for i in range(np.shape(alpha)[0]):
        result += alpha[i] * green_eval(point - grid[i, :], beta, k, d, M)
    return result
