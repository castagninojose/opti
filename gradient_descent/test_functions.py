"""Functions to test optimization routines."""
import numpy as np
from opti.gradient_descent.constants import GRADIENT_DESCENT_STEP_SIZE_METHODS
from opti.gradient_descent.optimize import gradient_descent_opt, newton_opt


def quadratic(x):
    return sum([(r) ** 2 for r in x])


if __name__ == "__main__":
    x_zero = np.array([331, 31])
    for m in GRADIENT_DESCENT_STEP_SIZE_METHODS:
        print(f"Method {m}: {gradient_descent_opt(quadratic, x_zero, method=m)}")
