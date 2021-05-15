import numpy as np
from scipy.constants import golden_ratio



def is_positive_definite(matrix):
    """Check if `matrix` is positive definite or not using cholesky factorization.

    Parameters
    ----------
    matrix : numpy.array
        Matrix to check

    Returns
    -------
        Bool indicating if `matrix` is positive definite.

    """
    if np.array_equal(matrix, matrix.T):
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def linear_golden_ratio(scalar_fun, epsilon=10**(-5), rho=1):
    """Exact linear search using the golden ratio method.

    Parameters
    ----------
    scalar_fun : function
        Linearization of the original function to optimize. i.e. `f(x + t * d)`
    epsilon : float
        Tolerance to stop iterations.
    rho : float
        Constant.
    
    Returns
    -------
    rv : float
        Minimizer for `scalar_fun`.

    """
    theta_1 = 1 / golden_ratio
    theta_2 = 1 - theta_1
    start = 0, middle = rho, stop = 2 * rho
    sacalar_fun_stop = scalar_fun(stop)
    sacalar_fun_middle = scalar_fun(middle)

    while sacalar_fun_stop < sacalar_fun_middle:
        start = middle, middle = stop, stop = 2 * stop
        sacalar_fun_middle = scalar_fun_stop
        scalar_fun_stop = scalar_fun(stop)

    first_node = start + theta_2 * (stop - start)
    second_node = start + theta_2 * (stop - start)
    scalar_fun_first = scalar_fun(first_node)
    scalar_fun_second = scalar_fun(second_node)

    while (stop - start) > epsilon:
        if scalar_fun_first < scalar_fun_second:
            stop = second_node
            second_node = first_node
            first_node = start + theta_1 * (stop - start)
            scalar_fun_second = scalar_fun_first
            scalar_fun_first = q(first_node)
        else:
            start = first_node
            first_node = second_node
            second_node = start + theta_2 * (start - stop)
            scalar_fun_first = scalar_fun_second
            scalar_fun_second = q(second_node)

    return  0.5 * (first_node + second_node)


def linear_armijo_rule(scalar_fun, direction, gamma=0.7, eta=0.45):
    """Parameters
    ----------
    scalar_fun : function
        Linearization of the original function to optimize. i.e. `f(x + t * d)`.
    direction : np.array
        Descent direction to minimize.
    gamma : float
        Constant.
    eta : float
        Constant.

    Returns
    -------
    rv : float
        Optimal step size that satisfies Arjimo's rule.

    """
    rv = 1
    while scalar_fun(t) > (scalar_fun(0) + eta * rv * np.inner(direction.T, direction)):
        rv = gamma * rv

    return rv
