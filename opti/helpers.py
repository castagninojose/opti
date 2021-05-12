import numpy as np


from opti.constants import (
    REACTION_POLICIES,
    MIN_INFECTION_RATE,
    LIFE_VALUE_COEFF_K,
    DISPO_COSTS_COEFF_M,
    DEFAULT_INFECTED_ZERO,
    DEFAULT_G_ZERO,
    DEFAULT_ALPHA,
    DEFAULT_THETA,
    DEFAULT_TOTAL_TIME,
)


def sir_model_simulator(
    infected_zero=DEFAULT_INFECTED_ZERO,
    g_zero=DEFAULT_G_ZERO,
    alpha=DEFAULT_ALPHA,
    theta=DEFAULT_THETA,
    reaction_policy=REACTION_POLICIES[0],
    coeff_k=LIFE_VALUE_COEFF_K,
    coeff_m=DISPO_COSTS_COEFF_M,
):

    infected = [infected_zero]
    removed = infected.copy()

    medical_costs = []
    dispo_costs = []

    assert (
        reaction_policy in REACTION_POLICIES
    ), f"Reaction policy must be one of {REACTION_POLICIES}. Instead got: {reaction_policy}."

    if reaction_policy == "short_sighted":
        transmition_rate = g_zero / (1 + theta * alpha * infected[0])
    else:
        transmition_rate = g_zero / (1 + theta * alpha * removed[0])

    for t in range(1, DEFAULT_TOTAL_TIME):
        new_infections = transmition_rate * infected[t - 1] * (1 - removed[t - 1])
        infected.append(new_infections)
        removed.append(removed[t - 1] + new_infections)

        if new_infections > MIN_INFECTION_RATE:
            medical_costs.append(coeff_k * infected[t])
            dispo_costs.append(coeff_m * (1 - transmition_rate / g_zero))

        if reaction_policy == "short_sighted":
            transmition_rate = g_zero / (1 + theta * alpha * infected[t])
        else:
            transmition_rate = g_zero / (1 + theta * alpha * removed[t])

    return {
        "infected": infected,
        "removed": removed,
        "dispo_costs": sum(dispo_costs),
        "medical_costs": sum(medical_costs),
    }


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
