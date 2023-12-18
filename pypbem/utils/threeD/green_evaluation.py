import numpy as np
from numba import jit


@jit(nopython=True)
def green_eval(point, k, d, M) -> np.complex64:
    """
    Evaluates the 3D Periodic Helmholtz Green Functions in the point (x, y) with
    incident wave perpendicular to the plane of spheres.

    Parameters
    ----------
    point : np.ndarray
        _size (1,3), point to evaluate_
    k : float
        _wavenumber_
    d : float
        _period of the problem_
    M : int
        _positive integer, number of terms in sum_

    Returns
    -------
    np.complex64
        _evaluation of the Green function_
    """
    # Assuming beta has the form [S, 0, 0], S a real number
    result = 0
    for n in range(-M, M + 1):
        for m in range(-M, M + 1):
            xi = np.array([0, n * d, m * d], dtype=np.float64)
            # e = np.exp(1j*np.inner(beta, xi))
            normX = np.linalg.norm(point - xi)
            if n == 0 and m == 0 and normX == 0:
                R = np.sqrt(np.sqrt(3) / np.pi) * 0.066 / 2
                result += (4 * np.pi) * (1 - np.exp(1j * k * R)) * (1j / (2 * k))
            else:
                result += np.exp(1j * k * normX) / (normX)

    result *= 1 / (4 * np.pi)
    return result
