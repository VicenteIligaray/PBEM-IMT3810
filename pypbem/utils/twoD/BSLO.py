from .green_evaluation import green_eval, green_eval2
import numpy as np


def assemble_BSLO(grid, d, k, beta, option=2) -> np.ndarray:
    """
    Assembles the 2D Boundary Single Layer Operator for a periodic Green function.

    Parameters
    ----------
    grid : np.ndarray
        _size(N,2), grid of circle_
    d : float
        _period of the problem_
    k : float
        _wvenumber_
    beta : float
        _k*sin(theta) where theta is the angle1 of the incident wave with the normal to
        the grating_
    option : 1 or 2
        _Whether to use the naive approach (1) to evaluate the Green function or
        to use the approach given by Linton 1998_

    Returns
    -------
    np.ndarray
        _Matrix of size (N,N) where N is the number of points in the grid_
    """
    M = 100

    N = np.shape(grid)[0]
    BSLO = np.zeros((N, N), dtype=np.complex128)
    if option == 1:
        for i in range(N):
            for j in range(N):
                if i != j:
                    BSLO[i, j] = green_eval(grid[i, :] - grid[j, :], beta, k, d, M)
                else:
                    # QUE PASA CUANDO Xi = Yj
                    if i != N - 1:
                        BSLO[i, j] = 0.5 * (
                            green_eval(grid[i + 1, :] - grid[j, :], beta, k, d, M)
                            + green_eval(grid[i, :] - grid[j + 1, :], beta, k, d, M)
                        )
                    else:
                        BSLO[i, j] = 0.5 * (
                            green_eval(grid[0, :] - grid[j, :], beta, k, d, M)
                            + green_eval(grid[i, :] - grid[0, :], beta, k, d, M)
                        )
    elif option == 2:
        for i in range(N):
            for j in range(N):
                if i != j:
                    BSLO[i, j] = green_eval2(grid[i, :] - grid[j, :], beta, k, d, M)
                else:
                    # QUE PASA CUANDO Xi = Yj
                    if i != N - 1:
                        BSLO[i, j] = 0.5 * (
                            green_eval2(grid[i + 1, :] - grid[j, :], beta, k, d, M)
                            + green_eval2(grid[i, :] - grid[j + 1, :], beta, k, d, M)
                        )
                    else:
                        BSLO[i, j] = 0.5 * (
                            green_eval2(grid[0, :] - grid[j, :], beta, k, d, M)
                            + green_eval2(grid[i, :] - grid[0, :], beta, k, d, M)
                        )
    return BSLO
