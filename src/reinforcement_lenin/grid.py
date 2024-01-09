import networkx as nx


class Board:
    def __init__(self, N: int = 4):
        BOARD_SIZE = N**2
        self.board: dict = {0: dict()}
        for k in range(1, BOARD_SIZE - 1):
            self.board[k] = {'up': k - N, 'down': k + N, 'left': k - 1, 'right': k + 1}
            if k - N < 0:
                self.board[k]['up'] = k
            if k + N > BOARD_SIZE - 1:
                self.board[k]['down'] = k
            if k % N == 0:
                self.board[k]['left'] = k
            if k % N == N - 1:
                self.board[k]['right'] = k
        self.board[BOARD_SIZE - 1] = dict()

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

    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)
