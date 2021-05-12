"""Collection of helper functions to numerically compute derivatives and differentials"""
import numpy as np


def is_positive_definite(matrix):
    """Check if `matrix` is positive definite or not using cholesky factorization.

    Parameters
    ----------
    matrix : numpy.array
        Matrix to check

    Returns
    -------
        Bool indicating if `matrix` is positive definite."""
    if np.array_equal(matrix, matrix.T):
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def finite_difference(callable_fun, x_zero, direction, diff_size=0.01):
    """Compute partial derivative of `callable_fun` in direction `direction`.

    Parameters
    ----------
    callable_fun : function
        Function to calculate the differential to.
    x_zero : int, float or array-type
        Point which the differential will be evaluated at
    direction : int
        Index for the coordinate to be used as derivative direction
    diff_size : float
        Step size to use for finite differences

    Returns
    -------
    rv : float
        Value of the derivative of `callable_fun` at `x_zero`
    """
    d = np.zeros(len(x_zero))
    d[direction] = 1.0
    forward = callable_fun(x_zero + diff_size * d)
    backward = callable_fun(x_zero - diff_size * d)
    rv = (forward - backward) / (2 * diff_size)

    return rv


def differential(callable_fun, x_zero, diff_size=0.01):
    """Compute differential of `callable_fun` at point `x_zero` using finite differences.

    Parameters
    ----------
    callable_fun : function
        Function to calculate the differential to.
    x_zero : int, float or array-type
        Point which the differential will be evaluated at
    diff_size : float
        Step size to use for finite differences

    Returns
    -------
    rv : numpy.array
        Value of the differential of `callable_fun` evaluated at `x_zero`
    """
    rv = []
    for direction in range(len(x_zero)):
        rv.append(
            finite_difference(callable_fun, x_zero, direction, diff_size=diff_size)
        )
    return np.array(rv)


def hessian(callable_fun, x_zero, diff_size=0.01):
    """Compute hessian matrix of `callable_fun`.

    Parameters
    ----------
    callable_fun : function
        Function to calculate the differential to.
    x_zero : int, float or array-type
        Point which the differential will be evaluated at
    diff_size : float
        Step size to use for finite differences

    Returns
    -------
    rv : numpy.array
        Hessian matrix of `callable_fun` evaluated at `x_zero`.
    """
    rv = []
    for direction in range(len(x_zero)):
        partial = lambda x: finite_difference(my_fun, x, direction, diff_size=diff_size)
        rv.append(differential(partial, x_zero, diff_size=diff_size))
    return np.array(rv)
