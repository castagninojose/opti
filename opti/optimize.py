import numpy as np
from opti.derivatives import gradient, hessian

DEFAULT_TOLERANCE = 0.001


def fixed_step_opt(
    callable_fun, start_point, step=DEFAULT_STEP_SIZE, tol=DEFAULT_TOLERANCE
):
    """Optimize `callable_fun` using a fixed step size the gradient's direction.

    """
    rv = start_point
    direction = -gradient(callable_fun, x_zero=start_point)
    while np.linalg.norm(direction) < tol:
        path = step * direction 
        rv = rv + path
        direction = -gradient(callable_fun, x_zero=rv)

    return rv


def newton_opt(callable_fun, start_point, tol=DEFAULT_TOLERANCE):
    """Optimize `callable_fun` using Newton-Raphson method."""
    rv = start_point
    direction = (-1) * -gradient(callable_fun, x_zero=start_point)
    while np.linalg.norm(direction) < tol:
        hessian = hessian(callable_fun, x_zero=rv)
        path = np.linalg.solve(hessian, direction)
        rv = rv + path
        direction = -gradient(callable_fun, x_zero=rv)
    
    return rv


def coordinate_opt(callable_fun, start_point)
