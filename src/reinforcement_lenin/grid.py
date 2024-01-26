from typing import Tuple

import click
import numpy as np
from networkx import Graph, draw, grid_2d_graph, relabel_nodes
from numpy.typing import ArrayLike, NDArray

from src.reinforcement_lenin.constants import (
    ACTIONS,
    DEFAULT_STATE_POLICY,
    OPTIMAL_POLICY,
)


class Board:
    """Representation of the board for a GridWorld game as described in Burton-Sutton,
    example 3.5 (p.60).

    Attributes
    ----------
    board_length : int
        Length of the board (i.e.) N.
    board_size : int
        Total size of the board (i.e.) N².
    non_terminals : list
        List of states that are not terminal.

    board: dict
        Dictionary representation for the legal actions in the game. Keys are states and
        values are the legal actions in such state in all directions also represented as
        a dictionary.
            {
                0: {'left': 0, 'up': 0, 'right': 1, 'down': 4},
                1: {'left': 0, 'up': 1, 'right': 2, 'down': 5},
                2: {'left': 1, 'up': 2, 'right': 3, 'down': 6},
                3: {'left': 2, 'up': 3, 'right': 4, 'down': 7},
                4: {'left': 4, 'up': 4, 'right': 5, 'down': 8},
                .
                .
                .
            }
    policy: numpy.NDArray
        Numpy array of size (`N`**2, 4) representing the probabilities for each action
        for each of the states. The first coordinate is the state and the second is the
        proba for the action, according with ACTIONS@constants.py.
            [
                [0.5, 0.5, 0, 0],
                [0.25, 0.25, 0.25, 0.25],
                .
                .
                .
                [0.25, 0.25, 0.25, 0.25],
                [0, 0, 0.5, 0.5],
            ]

    Methods
    -------
    TO DO !

    """

    def __init__(
        self,
        N: int = 4,
        reward: float = -1,
        discount_rate: float = 0.9,
        tolerance: float = 0.1,
        policy: str = 'random',
    ):
        """
        Parameters
        ----------
        N : int, default=4.
            Number of rows and columns for the board. Total nodes: N².

        """
        self.reward = reward
        self.discount_rate = discount_rate
        self.tol = tolerance
        self.board_length = N
        self.board_size = self.board_length**2

        if policy == 'random':
            self.policy = np.array(DEFAULT_STATE_POLICY * self.board_size)
            self.policy[0][:] = np.array([0.5, 0.5, 0, 0])
            self.policy[self.board_size - 1][:] = np.array([0, 0, 0.5, 0.5])
        elif policy == 'optimal':
            self.policy = OPTIMAL_POLICY
        else:
            raise ValueError(
                f"Select a valid policy. Expected 'optmial' or 'random', instead got {policy}."
            )

        self.board: dict = {0: dict()}
        for k in range(1, self.board_size - 1):
            self.board[k] = {'left': k - 1, 'up': k - N, 'right': k + 1, 'down': k + N}
            if k - N < 0:
                self.board[k]['up'] = k
            if k + N >= self.board_size:
                self.board[k]['down'] = k
            if k % N == 0:
                self.board[k]['left'] = k
            if k % N == N - 1:
                self.board[k]['right'] = k
        self.board[self.board_size - 1] = dict()

        self.non_terminals = list(self.board.keys())[1:-1]

    @property
    def graph(self) -> Graph:
        """Get network representation of the board according to the policy adpoted (i.e)
        the current state of `self.policy`. Takes no arguments.

        Returns
        -------
        networkx.Graph
            Graph representation constructed with nx.grid_2d_graph. Weights on the edge
            (u, v) represent the probability of moving from u to v.

        """
        rv = grid_2d_graph(self.board_length, self.board_length)
        rv = relabel_nodes(rv, {e: i + 1 for i, e in enumerate(rv.nodes)})
        for start, end in list(rv.edges):
            rv[start][end]["weight"] = self.policy[start][end % self.board_length]
        return rv

    @property
    def board_as_np(self) -> NDArray:
        """Get the board as numpy matrix. Useful to print. Takes no arguments.

        Returns
        -------
        np.array
            Matrix representation of the game board.

        """
        nodes = np.array(self.graph.nodes)
        return np.reshape(nodes, (self.board_length, self.board_length))

    def get_proba(
        self,
        state: int,
        future_state: int,
        action: str,
    ) -> float:
        """
        Compute the conditional probability of getting `reward` in `future_state` coming
        from `state` and performing `action`.

        Parameters
        ----------
        state: int
            Current state. Between 1 and N² - 2.
        future_state: int
            Future state. Between 1 and N² - 2.
        action: str
            One of 'left', 'up', 'right' or 'down' (see ACTIONS@constants.py).

        Returns
        -------
        float
            1 if taking `action` in `state` takes you to `future_state` and 0 otherwise.

        """
        if self.board[state][action] == future_state:
            return 1
        else:
            return 0

    def value_function(
        self,
        action: str,
        state: int,
        future_state: int,
        policy_value: ArrayLike,
    ) -> ArrayLike:
        """I am a function.

        Parameters
        ----------
        action : int
            Action to take.
        state : int
            Current state.
        future_state : int
            State to move to.
        policy_value : array-like, one-dimensional
            Array with current values.

        Returns
        -------
        array-like
        ??????????

        """
        proba = self.get_proba(state, future_state, action)
        return (policy_value[future_state] * self.discount_rate + self.reward) * proba

    def get_action_value_function(
        self,
        action: str,
        state: int,
        policy_value: ArrayLike,
    ) -> float:
        """Compute the expected return coming from `state`, taking `action` and
        following the current policy (`self.policy`)

        """
        rv = 0
        for future_state in self.board.keys():
            rv += self.value_function(action, state, future_state, policy_value)

        return rv

    def evaluate_policy(self) -> ArrayLike:
        """The expected return when starting from each state and following `self.policy`
        from then on. Based on Burton-Sutton, section 4.1 (p.75).

        Takes no arguments.

        Returns
        -------
        policy_value : numpy.ArrayLike
            ??????????????????????????

        """
        policy_value = np.zeros(len(self.board))
        delta = 1
        while delta > self.tol:
            delta = 0
            for state in self.non_terminals:
                v = policy_value[state]
                partial_sum = 0
                for action in ACTIONS.keys():
                    partial_sum += self.policy[state][
                        ACTIONS[action]
                    ] * self.get_action_value_function(action, state, policy_value)
                policy_value[state] = partial_sum

                delta = max(delta, abs(v - policy_value[state]))
        return policy_value

    def iterate_policy(self) -> Tuple[ArrayLike, NDArray]:
        """
        Iterate policy in search of that minimizes loss. Note that this will permanently
        modify `self.policy` attribute of the instance. Implemented following
        Burton-Sutton, section 4.3 (p.80).

        Takes no arguments.

        Returns
        -------
        rv : numpy.ArrayLike
            ????????????????
        self.policy : numpy.NDArray
            Modified policy.

        """
        while True:
            rv = self.evaluate_policy()
            policy_stable = True
            for state in self.non_terminals:
                old_action = self.policy[state].copy()
                q_actn_state = {
                    act: self.get_action_value_function(act, state, rv)
                    for act in ACTIONS.keys()
                }
                max_acts = [
                    k
                    for k, v in q_actn_state.items()
                    if v == max(q_actn_state.values())
                ]
                for action in ACTIONS.keys():
                    if action in max_acts:
                        self.policy[state][ACTIONS[action]] = 1 / len(max_acts)
                    else:
                        self.policy[state][ACTIONS[action]] = 0
                if not (old_action == self.policy[state]).all():
                    policy_stable = False

            # print(f"Reward hasta ahora: {rv}")
            # print(f"Politica hoy: \n{self.policy}")
            if policy_stable:
                return rv, self.policy
            else:
                continue

    def plot_graph(self) -> None:
        """
        Display the board using Networkx's default plotting engine. Takes no
        arguments and returns no value.
        """
        draw(self.graph, with_labels=True)


@click.command(context_settings={'show_default': True})
@click.option("--length", "-l", required=False, default=4, help="Board length.")
@click.option("--gamma", "-g", required=False, default=0.9, help="Discount rate.")
@click.option("--theta", "-t", required=False, default=0.1, help="Tolerance.")
@click.option("--reward", "-r", required=False, default=-1, help="Reward.")
@click.option(
    "--policy", "-p", required=False, default='random', help="Initial policy."
)
def main(length, gamma, theta, reward, policy):
    print(f"Lado del tabero {length}")
    print(f"Gamma (discount_rate) {gamma}")
    print(f"Theta (Tolerance) {theta}")
    print(f"Politica inicial {policy}")
    juego_1 = Board(
        N=length, policy=policy, tolerance=theta, discount_rate=gamma, reward=reward
    )
    _, policy1 = juego_1.iterate_policy()
    print(policy1)


if __name__ == "__main__":
    main()  # pylint:disable=E1120
