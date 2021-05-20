import numpy as np
from opti.constants import NEWTON_DEFAULT_NU, DEFAULT_TOLERANCE, MAX_ITER
from opti.derivatives import gradient, hessian
from opti.helpers import linear_golden_ratio, linear_armijo_rule, get_step_size


def gradient_descent_opt(
    objective_fun,
    start_point,
    tol=DEFAULT_TOLERANCE,
    method="numpymin",
):
    """Optimize `callable_fun` using a fixed step size the gradient's direction.

    Parameters
    ----------
    objective_fun : function
        Function to be optimized.
    start_point : numpy.array
        Starting point for the algorithm.
    tol : float
        Tolerance. If the gradient's norm falls bellow this value, algorithm terminates.
    method : str
        Method to use for linear search. One of `GRADIENT_DESCENT_STEP_SIZE_METHODS`.
    Returns
    -------
    rv : numpy.array
        Input that minimizes `callable_fun`.

    """
    rv = start_point
    direction = -gradient(objective_fun, x_zero=start_point)
    while np.linalg.norm(direction) > tol:
        step = get_step_size(objective_fun, direction, rv, method)
        path = step * direction
        rv = rv + path
        direction = -gradient(objective_fun, x_zero=rv)

    return rv


def newton_opt(objective_fun, start_point, tol=DEFAULT_TOLERANCE):
    """Optimize `objective_fun` using Newton-Raphson method.

    Parameters
    ----------
    objective_fun : function
        Function to be optimized.
    start_point : numpy.array
        Starting point for the algorithm.
    tol : float
        Tolerance. If the gradient's norm falls bellow this value, algorithm terminates.

    Returns
    -------
    rv : numpy.array
        Input that minimizes `objective_fun`.

    """
    rv = start_point
    iter = 0
    direction = -gradient(objective_fun, x_zero=start_point)
    while (np.linalg.norm(direction) < tol) & (iter < MAX_ITER):
        iter += 1
        hessian = hessian(objective_fun, x_zero=rv)
        if not is_positive_definite(hessian):
            hessian = hessian + NEWTON_DEFAULT_NU * np.identity(len(start_point))
        path = np.linalg.solve(hessian, direction)
        rv = rv + path
        direction = -gradient(objective_fun, x_zero=rv)

    return rv
