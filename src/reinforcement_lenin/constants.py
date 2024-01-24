from numpy import array

ACTIONS = {"left": 0, "up": 1, "right": 2, "down": 3}
DEFAULT_STATE_POLICY = [[0.25] * 4]
OPTIMAL_POLICY = array(
    [
        [0.5, 0.5, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0.5, 0, 0, 0.5],
        [0, 1, 0, 0],
        [0.5, 0.5, 0, 0],
        [0.5, 0, 0, 0.5],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0.5, 0.5, 0],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1],
        [0, 0.5, 0.5, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0.5, 0.5],
    ]
)
