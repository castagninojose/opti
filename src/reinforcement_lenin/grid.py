import networkx as nx
import numpy as np


class Board:
    def __init__(self, N: int = 4):
        self.board_size = N**2
        self.board: dict = {0: dict()}
        for k in range(1, self.board_size - 1):
            self.board[k] = {'up': k - N, 'down': k + N, 'left': k - 1, 'right': k + 1}
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

    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)
