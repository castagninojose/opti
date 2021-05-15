DEFAULT_STEP_SIZE = 0.01
DEFAULT_TOLERANCE = 0.001
NEWTON_DEFAULT_NU = 10
GRADIENT_DESCENT_STEP_SIZE_METHODS = ["fixed_size", "numpymin", "armijo"]
GRADIENT_DESCENT_DEFAULT_KWARGS = {
    "step_size": DEFAULT_STEP_SIZE,
    "npmin_left": 0,
    "npmin_right": 10,
}
