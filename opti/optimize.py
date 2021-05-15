import numpy as np
from scipy import optimize
from opti.constants import NEWTON_DEFAULT_NU, DEFAULT_STEP_SIZE
from opti.derivatives import gradient, hessian
from opti.helpers import linear_golden_ratio, linear_armijo_rule


def get_step_size(
    callable_fun,
    direction,
    eval_point,
    method,
):
    assert (
        method in GRADIENT_DESCENT_STEP_SIZE_METHODS
    ), f"Step size method must be one of {GRADIENT_DESCENT_STEP_SIZE_METHODS}. Instead got {method}."

    q = lambda t: callable_fun(eval_point + t * direction)

    if method == "fixed_size":
        return DEFAULT_STEP_SIZE
    elif method == "numpymin":
        return optimize.fminbound(q, 0, 10)
    elif method == "golden_ratio":
        return linear_golden_ratio(q)
    elif method == "armijo":
        return linear_armijo_rule(q, direction)


def gradient_descent_opt(
    callable_fun,
    start_point,
    step=DEFAULT_STEP_SIZE,
    tol=DEFAULT_TOLERANCE,
    method="numpymin",
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
        step = get_step_size(callable_fun, direction, rv, method)
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
