import networkx as nx
import numpy as np

from src.reinforcement_lenin.constants import ACTIONS


class Board:
    def __init__(self, N: int = 4):
        self.board_size = N**2
        self.board: dict = {0: dict()}
        self.policy: dict = {1: dict()}
        for k in range(1, self.board_size - 1):
            self.board[k] = {'left': k - 1, 'up': k - N, 'right': k + 1, 'down': k + N}
            self.policy[k] = {'left': 0.25, 'up': 0.25, 'right': 0.25, 'down': 0.25}
            if k - N < 0:
                self.board[k]['up'] = k
            if k + N > self.board_size - 1:
                self.board[k]['down'] = k
            if k % N == 0:
                self.board[k]['left'] = k
            if k % N == N - 1:
                self.board[k]['right'] = k
        self.board[self.board_size - 1] = dict()

        self.graph = nx.DiGraph()
        for k, n in self.board.items():
            for direction in n.values():
                self.graph.add_edge(k, direction)

        self.non_terminals = sorted(list(self.graph.nodes))[1:-1]

    def get_proba(
        self, state, future_state, action, reward=-1
    ):  # pylint: disable=unused-argument
        if self.board[state][action] == future_state:
            return 1
        else:
            return 0

    def as_numpy(self):
        nodes = np.array(self.graph.nodes)
        return np.reshape(nodes, (self.board_size, self.board_size))

    def value_function(self, action, state, future_state, reward, discount_rate):
        proba = self.get_proba(state, future_state, action, reward=reward)
        return (self.policy[future_state][action] * discount_rate + reward) * proba

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
                            for f in self.non_terminals
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

    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)


if __name__ == "__main__":
    juego_1 = Board()
    print(juego_1.evaluate_policy(0.5, -1, 0.5))
