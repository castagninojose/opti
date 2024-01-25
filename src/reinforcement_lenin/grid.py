import click
import networkx as nx
import numpy as np

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
    # TODO: documentar mejor esta parte, los nodos terminales les falta info
    policy: dict
        Similar to `board`, the policy is represented as a dict and this time the values
        represent probabilities instead of legal moves.
            {
                1: {'left': 0.25, 'up': 0.25, 'right': 0.25, 'down': 0.25},
                2: {'left': 0.25, 'up': 0.25, 'right': 0.25, 'down': 0.25},
                .
                .
                .
            }
    graph: nx.DiGraph
        Graph representation, useful to visualize, simulate and navigate the board.
    non_terminals : list
        List of states that are not terminal (e.g. 0 or N² - 1).

    """

    def __init__(self, N: int = 4, policy: str = 'random'):
        """
        Parameters
        ----------
        N : int, default=4.
            Number of rows and columns for the board. Total nodes: N².

        """
        self.board_size = N**2

        if policy == 'random':
            self.policy = np.array(DEFAULT_STATE_POLICY * self.board_size)
            self.policy[0][:] = np.array([0.5, 0.5, 0, 0])
            self.policy[self.board_size - 1][:] = np.array([0, 0, 0.5, 0.5])
        elif policy == 'optimal':
            self.policy = OPTIMAL_POLICY

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

        self.graph = nx.DiGraph()
        for k, n in self.board.items():
            for direction in n.values():
                self.graph.add_edge(k, direction)

        # for k, n in self.board.items():
        #     for direction in n.values():
        #         self.graph.add_edge(k, n[direction])
        #         self.graph.edges[k, n[direction]]["weight"] = self.policy[k][direction]

    def get_proba(
        self, state, future_state, action, reward=-1
    ):  # pylint: disable=unused-argument
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
        reward: float
            Expected reward given after `action` in `state`.

        Returns
        -------
        float
            Either 1 or 0 for now =p

        """
        if self.board[state][action] == future_state:
            return 1
        else:
            return 0

    def as_numpy(self):
        """Represent the board in numpy format. Takes no arguments.

        Returns
        -------
        np.array
            Matrix representation of the game board.

        """
        nodes = np.array(self.graph.nodes)
        return np.reshape(nodes, (self.board_size, self.board_size))

    def value_function(
        self, action, state, future_state, reward, discount_rate, policy_value
    ):
        """I am a function. That's all I know.

        Parameters
        ----------
        action : int
            Action to take.
        state : int
            Current state.
        future_state : int
            State to move to.
        reward : float
            Reward set for the game.
        discount_rate : float
            Discount rate, or gamma.
        policy_value : array-like, one-dimensional
            Array with current values.

        Returns
        -------
        array-like
        """
        proba = self.get_proba(state, future_state, action, reward=reward)
        return (policy_value[future_state] * discount_rate + reward) * proba

    def get_action_value_function(
        self, action, state, reward, discount_rate, policy_value
    ):
        """Compute the expected return coming from `state`, taking `action` and
        following the current policy (`self.policy`)

        """
        rv = 0
        for future_state in self.board.keys():
            rv += self.value_function(
                action, state, future_state, reward, discount_rate, policy_value
            )

        return rv

    def evaluate_policy(self, discount_rate: float, reward: float, tolerance: float):
        """The expected return when starting from each `state` and following
        `self.policy` from then on.

        """
        policy_value = np.zeros(len(self.board))
        delta = 1
        while delta > tolerance:
            delta = 0
            for state in self.non_terminals:
                v = policy_value[state]
                partial_sum = 0
                for action in ACTIONS.keys():
                    partial_sum += self.policy[state][
                        ACTIONS[action]
                    ] * self.get_action_value_function(
                        action, state, reward, discount_rate, policy_value
                    )
                policy_value[state] = partial_sum

                delta = max(delta, abs(v - policy_value[state]))
        return policy_value

    def iterate_policy(self, discount_rate: float, reward: float, tolerance: float):
        while True:
            rv = self.evaluate_policy(discount_rate, reward, tolerance)
            policy_stable = True
            for state in self.non_terminals:
                old_action = self.policy[state].copy()
                q_actn_state = {
                    act: self.get_action_value_function(
                        act, state, reward, discount_rate, rv
                    )
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

    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)


@click.command()
@click.option("--length", "-l", required=True, default=4, help="Board length.")
@click.option("--gamma", "-g", required=False, default=0.9, help="Discount rate.")
@click.option("--theta", "-t", required=False, default=0.1, help="Tolerance.")
@click.option(
    "--policy", "-p", required=False, default='random', help="Initial policy."
)
def main(length, gamma, theta, reward, policy):
    juego_1 = Board(length, policy)
    # aver = juego_1.evaluate_policy(gamma, -1, theta)
    _, policy1 = juego_1.iterate_policy(gamma, reward, theta)
    print(policy1)


if __name__ == "__main__":
    main()  # pylint:disable=E1120
