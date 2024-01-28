from numpy import array

ACTIONS = {"left": 0, "up": 1, "right": 2, "down": 3}

DEFAULT_STATE_POLICY = [[0.25] * 4]

OPTIMAL_POLICY = array(
    [  # optimal policy for a Grid World game of 4 by 4
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
OPTIMAL_POLICY_3 = array(
    [  # optimal policy for a Grid World game of 3 by 3
        [0.5, 0.5, 0, 0],
        [1, 0, 0, 0],
        [0.5, 0, 0, 0.5],
        [0, 1, 0, 0],
        [0.25, 0.25, 0.25, 0.25],
        [0, 0, 0, 1],
        [0, 0.5, 0.5, 0],
        [0, 0, 1, 0],
        [0, 0, 0.5, 0.5],
    ]
)
