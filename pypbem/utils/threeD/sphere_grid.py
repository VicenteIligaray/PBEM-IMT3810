import numpy as np
import bempp.api


def get_sphere_grid(h) -> np.ndarray:
    """
    Creates grid for unit sphere.

    Parameters
    ----------
    h : float
        _see description of bempp.api.shapes.sphere_

    Returns
    -------
    np.ndarray
        _size(N,3) points of unit sphere_
    """
    if not isinstance(h, float) and not isinstance(h, int):
        raise ValueError("h must be of type float or int")

    if h <= 0:
        raise ValueError("h must be positive")

    grid = bempp.api.shapes.sphere(r=1, origin=(0, 0, 0), h=h)

    return grid.centroids
