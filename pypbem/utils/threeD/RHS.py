import numpy as np


def assemble_RHS(grid, k) -> np.ndarray:
    """
    Assembles the 3D right hand side by approximating the integral by evaluating
    in the point.

    Parameters
    ----------
    grid : np.ndarray
        _size (N,3), grid of the sphere_
    k : float
        _wavenumber_

    Returns
    -------
    np.ndarray
        _size (N,) where N is the number of points in the grid_
    """

    N = np.shape(grid)[0]
    RHS = np.zeros((N,), dtype=np.complex64)
    for i in range(N):
        RHS[i] = np.exp(1j * k * grid[i, 0])
    return RHS
