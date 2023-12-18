from scipy.special import hankel1, spence
import numpy as np
from cmath import sqrt


def green_eval(point, beta, k, d, M) -> np.complex64:
    """
    Evaluates the 2D Periodic Helmholtz Green Function in the point (x, y)
    with the naive approach.

    Parameters
    ----------
    point : np.ndarray
        _size (1,2), point to evaluate_
    beta : float
        _k*sin(theta) where theta is the angle1 of the incident wave with the normal to
        the grating_
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
    x, y = point[:]
    x_squared = x**2
    ibetad = 1j * beta * d
    result = hankel1(0, k * sqrt(x_squared + y**2))

    for m in range(1, M + 1):
        # positive part
        rm = sqrt(x_squared + (y - m * d) ** 2)
        e = np.exp(m * ibetad)
        result += hankel1(0, k * rm) * e

        # negative part
        rm = sqrt(x_squared + (y + m * d) ** 2)
        e_inv = e ** (-1)
        result += hankel1(0, k * rm) * e_inv

    result *= -1j / 4
    result *= np.exp(-1j * beta * y)
    return result


def green_eval2(point, beta, k, d, M) -> np.complex64:
    """
    Evaluates the Periodic Helmholtz Green Function in the point (x, y)
    with the approach given by Linton 1998 by Kummer's transformation.

    Parameters
    ----------
    point : np.ndarray
        _size (1,2), point to evaluate_
    beta : float
        _k*sin(theta) where theta is the angle1 of the incident wave with the normal to
        the grating_
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
    p = 2 * np.pi / d
    x, y = point[:]
    gammazero = -1j * sqrt(k**2 - beta**2)
    result = np.exp(-gammazero * np.abs(x)) / gammazero

    # Calculate S
    z = np.abs(x) + 1j * y
    if z == 1 + 0j:
        result += 1
    else:
        zconj = np.abs(x) - 1j * y
        l1z = -np.log(1 - np.exp(-p * z))
        l1zconj = -np.log(1 - np.exp(-p * zconj))
        l2z = spence(1 - np.exp(-p * z))
        l2zconj = spence(1 - np.exp(-p * zconj))
        tempzconj = l1zconj / p - l2zconj * (2 * beta - k**2 * np.abs(x)) / (2 * p * p)
        tempzconj *= np.exp(-beta * np.abs(x))
        tempz = l1z / p + l2z * (2 * beta + k**2 * np.abs(x)) / (2 * p * p)
        tempz *= np.exp(beta * np.abs(x))

        result += tempz + tempzconj

    for m in range(1, M + 1):
        pm = p * m
        # positive part
        betam = beta + pm
        gammam = -1j * sqrt(k**2 - betam**2)
        um = (1 - beta / pm + k**2 * np.abs(x) / (2 * pm)) * np.exp(-(pm + beta) * np.abs(x)) / pm
        result += (np.exp(-gammam * np.abs(x)) / gammam - um) * np.exp(1j * pm * y)

        # negative part
        betam = beta - pm
        gammam = -1j * sqrt(k**2 - betam**2)
        um = (1 + beta / pm + k**2 * np.abs(x) / (2 * pm)) * np.exp(-(pm - beta) * np.abs(x)) / pm
        result += (np.exp(-gammam * np.abs(x)) / gammam - um) * np.exp(-1j * pm * y)
    result *= -np.exp(1j * beta * y) / (2 * d)
    result *= np.exp(-1j * beta * y)
    return result
