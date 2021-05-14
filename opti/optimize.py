import numpy as np
from opti.constants import NEWTON_DEFAULT_NU
from opti.derivatives import gradient, hessian


def fixed_step_opt(
    callable_fun, start_point, step=DEFAULT_STEP_SIZE, tol=DEFAULT_TOLERANCE
):
    """Optimize `callable_fun` using a fixed step size the gradient's direction.

    Parameters
    ----------
    callable_fun : function
        Function to be optimized.
    start_point : numpy.array
        Starting point for the algorithm.
    step : float
        Step size.
    tol : float
        Tolerance. If the gradient's norm falls bellow this value, algorithm terminates.

    Returns
    -------
    rv : numpy.array
        Input that minimizes `callable_fun`.

    """
    rv = start_point
    direction = -gradient(callable_fun, x_zero=start_point)
    while np.linalg.norm(direction) < tol:
        path = step * direction
        rv = rv + path
        direction = -gradient(callable_fun, x_zero=rv)

    return rv


def newton_opt(callable_fun, start_point, tol=DEFAULT_TOLERANCE):
    """Optimize `callable_fun` using Newton-Raphson method.

    Parameters
    ----------
    callable_fun : function
        Function to be optimized.
    start_point : numpy.array
        Starting point for the algorithm.
    tol : float
        Tolerance. If the gradient's norm falls bellow this value, algorithm terminates.

    Returns
    -------
    rv : numpy.array
        Input that minimizes `callable_fun`.

    """
    rv = start_point
    direction = -gradient(callable_fun, x_zero=start_point)
    while np.linalg.norm(direction) < tol:
        hessian = hessian(callable_fun, x_zero=rv)
        if not is_positive_definite(hessian):
            hessian = hessian + NEWTON_DEFAULT_NU * np.identity(len(start_point))
        path = np.linalg.solve(hessian, direction)
        rv = rv + path
        direction = -gradient(callable_fun, x_zero=rv)

    return rv
