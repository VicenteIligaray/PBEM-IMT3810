import numpy as np


def get_circle_grid(h) -> np.ndarray:
    """
    Creates grid for unit circle.

    Parameters
    ----------
    h : int
        _1/h the number of point in the grid_

    Returns
    -------
    np.ndarray
        _size (1/h, 2), the points are equispaced_
    """
    if not isinstance(h, float) and not isinstance(h, int):
        raise ValueError("h must be of type float or int")

    if h <= 0:
        raise ValueError("h must be positive")

    N = int(1 / h)
    theta = np.linspace(0, 2 * np.pi, N)[0:-1]
    x = np.cos(theta)
    y = np.sin(theta)
    return np.array([x, y]).T
