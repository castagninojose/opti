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
