import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



class Juego:
    def __init__(self, N: int = 4):
        BOARD_SIZE = N ** 2
        self.board_dict = {0: dict()}
        for k in range(1, BOARD_SIZE - 1):
            self.board_dict[k] = (
                {'up': k - N, 'down': k + N, 'left': k - 1, 'right': k + 1}
            )
            if k - N < 0:
                self.board_dict[k]['up'] = k
            if k + N > BOARD_SIZE - 1:
                self.board_dict[k]['down'] = k
            if k % N == 0:
                self.board_dict[k]['left'] = k
            if k % N == N - 1:
                self.board_dict[k]['right'] = k
        self.board_dict[BOARD_SIZE - 1] = dict()

        self.graph = nx.DiGraph()
        for k, n in self.board_dict.items():
            for direction in n.values():
                self.graph.add_edge(k, direction)

    def get_proba(self, state, future_state, action, reward=-1):
        if self.board_dict[state][action] == future_state:
            return 1
        else:
            return 0

    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)

