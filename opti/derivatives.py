"""Collection of helper functions to numerically compute derivatives and differentials"""
import numpy as np
from opti.constants import DEFAULT_STEP_SIZE


def finite_difference(callable_fun, x_zero, direction, step=DEFAULT_STEP_SIZE):
    """Compute partial derivative of `callable_fun` in direction `direction`.

    Parameters
    ----------
    callable_fun : function
        Function to calculate the gradient to.
    x_zero : int, float or array-type
        Point which the differential will be evaluated at
    direction : int
        Index for the coordinate to be used as derivative direction
    step : float
        Step size to use for finite differences

    Returns
    -------
    rv : float
        Value of the derivative of `callable_fun` at `x_zero`

    """
    d = np.zeros(len(x_zero))
    d[direction] = 1.0
    forward = callable_fun(x_zero + step * d)
    backward = callable_fun(x_zero - step * d)
    rv = (forward - backward) / (2 * step)

    return rv


def gradient(callable_fun, x_zero, step=DEFAULT_STEP_SIZE):
    """Compute gradient of `callable_fun` at point `x_zero` using finite differences.

    Parameters
    ----------
    callable_fun : function
        Function to calculate the gradient to.
    x_zero : int, float or array-type
        Point which the gradient will be evaluated at
    step : float
        Step size to use for finite differences

    Returns
    -------
    rv : numpy.array
        Value of the gradient of `callable_fun` evaluated at `x_zero`

    """
    rv = []
    for direction in range(len(x_zero)):
        rv.append(finite_difference(callable_fun, x_zero, direction, step=step))
    return np.array(rv)


def hessian(callable_fun, x_zero, step=DEFAULT_STEP_SIZE):
    """Compute hessian matrix of `callable_fun`.

    Parameters
    ----------
    callable_fun : function
        Function to calculate the gradient to.
    x_zero : int, float or array-type
        Point which the gradient will be evaluated at
    step : float
        Step size to use for finite differences

    Returns
    -------
    rv : numpy.array
        Hessian matrix of `callable_fun` evaluated at `x_zero`.

    """
    rv = []
    for direction in range(len(x_zero)):
        partial = lambda x: finite_difference(callable_fun, x, direction, step=step)
        rv.append(gradient(partial, x_zero, step=step))
    return np.array(rv)
