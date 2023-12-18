import numpy as np
from .green_evaluation import green_eval, green_eval2


def potential_eval(point, grid, alpha, d, k, beta, option=2) -> np.complex64:
    """
    Evaluates the 2D single layer potential operator by approximating the
    integral, evaluating at the point the Green function

    Parameters
    ----------
    point : np.ndarray
        _size (1,2), point to evaluate_
    grid : np.ndarray
        _size (N,2), grid of the circle_
    alpha : np.ndarray
        _size (N,) coefficients of scattered field_
    d : float
        _period of the problem_
    k : float
        _wavenumber_
    beta : float
        _k*sin(theta) where theta is the angle1 of the incident wave with the normal to
        the grating_
    option : 1 or 2
        _Whether to use the naive approach (1) to evaluate the Green function or
        to use the approach given by Linton 1998_

    Returns
    -------
    np.complex64
        _evaluation of the operator_
    """
    M = 100

    result = 0
    if option == 1:
        for i in range(np.shape(alpha)[0]):
            result += alpha[i] * green_eval(point - grid[i, :], beta, k, d, M)
    elif option == 2:
        for i in range(np.shape(alpha)[0]):
            result += alpha[i] * green_eval2(point - grid[i, :], beta, k, d, M)
    return result
