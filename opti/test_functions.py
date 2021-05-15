"""Functions to test optimization routines."""
import numpy as np
from opti.optimize import gradient_descent_opt


def quadratic(x):
    return sum([(r) ** 2 for r in x])


if __name__ == "__main__":
    x_zero = np.array([331, 31])
    print(gradient_descent_opt(quadratic, x_zero))
    print(gradient_descent_opt(quadratic, x_zero, method="fixed_size"))
    print(gradient_descent_opt(quadratic, x_zero, method="golden_ratio"))
    print(gradient_descent_opt(quadratic, x_zero, method="armijo"))
    print(newton_opt(quadratic, x_zero))
