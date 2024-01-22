import networkx as nx
import numpy as np

from src.reinforcement_lenin.constants import ACTIONS, DEFAULT_POLICY


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

    def __init__(self, N: int = 4):
        """
        Parameters
        ----------
        N : int, default=4.
            Number of rows and columns for the board. Total nodes: N².

        """
        self.board_size = N**2
        self.board: dict = {0: dict()}
        self.policy: dict = {0: {'left': 0.5, 'up': 0.5, 'right': 0, 'down': 0}}
        for k in range(1, self.board_size - 1):
            self.policy[k] = DEFAULT_POLICY
            self.board[k] = {'left': k - 1, 'up': k - N, 'right': k + 1, 'down': k + N}
            if k - N < 0:
                self.board[k]['up'] = k
            if k + N >= self.board_size - 1:
                self.board[k]['down'] = k
            if k % N == 0:
                self.board[k]['left'] = k
            if k % N == N - 1:
                self.board[k]['right'] = k
        self.board[self.board_size - 1] = dict()
        self.policy[self.board_size - 1] = {
            'left': 0,
            'up': 0,
            'right': 0.5,
            'down': 0.5,
        }

        self.graph = nx.DiGraph()
        for k, n in self.board.items():
            for direction in n.values():
                self.graph.add_edge(k, direction)

        self.non_terminals = list(self.board.keys())[1:-1]

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

    def value_function(self, action, state, future_state, reward, discount_rate):
        proba = self.get_proba(state, future_state, action, reward=reward)
        return (self.policy[future_state][action] * discount_rate + reward) * proba

    def get_action_value_function(self, action, state, reward, discount_rate):
        rv = 0
        for future_state in self.board.keys():
            rv += self.value_function(
                action, state, future_state, reward, discount_rate
            )

        return rv

    def evaluate_policy(self, discount_rate: float, reward: float, tolerance: float):
        policy_value = np.zeros(len(self.board))
        delta = 1
        while delta > tolerance:
            delta = 0
            for state in self.non_terminals:
                v = policy_value[state]
                for action in ACTIONS:
                    expected_value = sum(
                        [
                            self.value_function(action, state, f, reward, discount_rate)
                            for f in self.board.keys()
                        ]
                    )
                    partial_sum = sum(
                        [
                            self.policy[state][action] * expected_value
                            for action in ACTIONS
                        ]
                    )
                policy_value[state] = partial_sum

                delta = max(delta, abs(v - policy_value[state]))
        return policy_value

    def iterate_policy(self, discount_rate: float, reward: float, tolerance: float):
        # rv = self.evaluate_policy(discount_rate, reward, tolerance)
        # policy_stable = True
        # for state in self.non_terminals:
        #     old_action = self.policy[state]
        # self.policy[]
        pass

    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)


if __name__ == "__main__":
    juego_1 = Board()
    for gamma in [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]:
        print(f"ESTO ES GAMMA: {gamma}")
        print(juego_1.evaluate_policy(gamma, -1, 0.1))
    # juego_1.plot_graph()
