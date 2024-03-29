from typing import List, Tuple, Union

import click
import numpy as np
from networkx import (  # shortest_path,
    Graph,
    draw_networkx_edges,
    draw_networkx_labels,
    draw_networkx_nodes,
    grid_2d_graph,
    relabel_nodes,
)
from numpy.typing import ArrayLike, NDArray
from pyvis import network as pyvisnet

from src.reinforcement_lenin.constants import (
    ACTIONS,
    BURTON_POLICY,
    DEFAULT_STATE_POLICY,
)


class Board:
    """Representation of the board for a GridWorld game as described in Burton-Sutton,
    example 3.5 (p.60).

    Attributes
    ----------
    board_length : int
        Length of the board (i.e.) `N`.
    board_size : int
        Total size of the board (i.e.) `N`².
    terminals : list of int, default=[0, `N`² - 1]
        Terminal states.
    non_terminals : list of int, default=[1 : `N`² - 2]
        List of states that are not terminal.
    tol : float, default=0.01

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
        Numpy array of size (`N`², 4) representing the probabilities for each action
        for each of the states. The first coordinate is the state and the second is the
        proba for the action, indexed according to ACTIONS@constants.py. Default is
            [
                [0.5, 0.5, 0, 0],
                [0.25, 0.25, 0.25, 0.25],
                .
                .
                .
                [0.25, 0.25, 0.25, 0.25],
                [0, 0, 0.5, 0.5],
            ]
    reward : numpy.NDarray
        Numpy array of size (`N`², 4) representing the reward for each action for each of
        the states. The first coordinate is the state and the second is the reward indexed
        according to ACTIONS@constants.py. Default is
            [
                [-1, -1, -1, -1],
                .
                .
                .
                [-1, -1, -1, -1],
            ]
    graph : networkx.Graph
        Graph representation of the state of `self.policy`. Used for visualizations only.
    board_as_np : numpy.NDArray
        Board represented as numpy array of size (`N`², `N`). (deprecated)

    """

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        N: int = 4,
        reward: ArrayLike = None,
        discount_rate: float = 0.9,
        tolerance: float = 0.01,
        default_policy: str = 'random',
        terminals: List[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        N : int, default=4.
            Number of rows (and columns) for the board. Total states: `N`². Boards with
            two or less rows are considered trivial and ValueError will be raised if
            input is less than 3.
        reward : array-like, default=([-1, -1, -1, -1] * `N`²)
            Array of dimensions (`N`², 4) with the reward(s) set for the game.
        discount_rate : float, default=0.9
            Discount rate set for the agent, referred to as gamma in Burton-Sutton. It
            must be at least 0 and at most 1.
        tolerance : float, default=0.01
            Tolerance. Any improvement beyond this limit will result in a stall in the
            policy iteration.
        policy : str, default='random'
            Default policy to adopt. If 'random'
        terminals : list of int, default=None (first and last node only)
            Nodes to act as terminals for the game. Must be between 0 and `N - 1`.

        """
        if N < 3:
            raise ValueError("Board must be at least length `N` = 3.")

        # set board general properties
        self.board_length: int = N
        self.board_size: int = self.board_length**2
        self.discount_rate: float = discount_rate
        self.tol: float = tolerance
        if reward is None:
            self.reward = np.array([[-1] * 4] * self.board_size)
        else:
            # TODO: que el usuario pueda editar valores individuales
            self.reward = reward

        self._states: List[int] = list(range(self.board_size))
        if terminals is None:
            self.terminals = [0, self.board_size - 1]
        elif isinstance(terminals, int):
            self.terminals = [terminals]
        else:
            self.terminals = terminals

        self.non_terminals = list(set(self._states) - set(self.terminals))

        # set boar attribute
        self.board: dict = {}
        for n in self.terminals:
            self.board[n] = dict()
        for n in self.non_terminals:
            self.board[n] = {'left': n - 1, 'up': n - N, 'right': n + 1, 'down': n + N}
            if n - N < 0:
                self.board[n]['up'] = n
            if n + N >= self.board_size:
                self.board[n]['down'] = n
            if n % N == 0:
                self.board[n]['left'] = n
            if n % N == N - 1:
                self.board[n]['right'] = n

        # set policy attribute
        if default_policy not in ['burton', 'random']:
            raise ValueError(f"Invalid default policy {default_policy}.")

        if default_policy == 'random':
            self.policy = np.array(DEFAULT_STATE_POLICY * self.board_size)
        if default_policy == 'burton':
            self.policy = BURTON_POLICY

    @property
    def graph(self) -> Graph:
        """Get network representation of the board according to the policy adpoted (i.e)
        the current state of `self.policy`.

        The second index (column) used to get values from `self.policy` is retrieved using
        inverse lookup on `self.board[state]`, which is a one-to-one map for all states.

        Takes no arguments.

        Returns
        -------
        networkx.Graph
            Graph representation constructed with nx.grid_2d_graph. Weights on the edge
            (u, v) represent the probability of moving from u to v.

        """
        rv = grid_2d_graph(self.board_length, self.board_length)
        rv = relabel_nodes(rv, {node: ix for ix, node in enumerate(rv.nodes)})
        rv = rv.to_directed()
        # fill terminal nodes policy manually
        for t in self.terminals:
            for n in rv.neighbors(t):
                rv[t][n]["weight"] = 0
        # the rest is filled using `self.policy`
        for start in self.non_terminals:
            for end in rv.neighbors(start):
                end_ix = int(
                    list(self.board[start].values()).index(end)
                )  # cast as int cuz of reasons
                rv[start][end]["weight"] = self.policy[start][end_ix]

        return rv

    def get_proba(
        self,
        state: int,
        future_state: int,
        action: str,
        reward: float,
        constant_reward: bool = True,
    ) -> float:
        """
        Compute the conditional probability of getting `reward` in `future_state` coming
        from `state` and performing `action`.

        Parameters
        ----------
        state : int
            Current state. Between 1 and `N`² - 2.
        future_state: int
            Future state. Between 1 and `N`² - 2.
        action : str
            One of 'left', 'up', 'right' or 'down' (see ACTIONS@constants.py).
        reward : float
            Reward set for the game.
        constant_reward : bool, default=True
            Boolean indicating if the reward is a constant value or not.

        Returns
        -------
        float
            1 if taking `action` in `state` takes you to `future_state` and 0 otherwise.

        """
        legal_action = self.board[state][action] == future_state
        valid_reward = self.reward[state][ACTIONS[action]] == reward

        if constant_reward:
            if legal_action:
                return 1
            else:
                return 0
        else:
            if legal_action and valid_reward:
                return 1
            else:
                return 0

    def value_function(
        self,
        action: str,
        state: int,
        future_state: int,
        policy_value: ArrayLike,
        reward: float,
    ) -> float:
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
        reward : float
            Reward set for the game.

        Returns
        -------
        float
            Function value.

        """
        proba = self.get_proba(state, future_state, action, reward)
        return (policy_value[future_state] * self.discount_rate + reward) * proba

    def get_action_value_function(
        self,
        action: str,
        state: int,
        policy_value: ArrayLike,
    ) -> float:
        """Compute expected return coming from `state`, taking `action` and following
        policy (`self.policy`).

        Returns the sum of of `self.value_function(action, state future_state)`, over all
        future_states. Used in `self.evaluate_policy` and `selfl.iterate_policy`.

        Parameters
        ----------
        action : str,
            Action.
        state : int,
            Source state.
        policy_value : array-like
            Current policy values.

        Returns
        -------
        rv : float
            The specified sum.
        """
        rv = 0.0
        for future_state in self.board.keys():
            rv += self.value_function(
                action,
                state,
                future_state,
                policy_value,
                self.reward[state][ACTIONS[action]],
            )

        return rv

    def evaluate_policy(self) -> ArrayLike:
        """The expected return when starting from each state and following `self.policy`
        from then on. Based on Burton-Sutton, section 4.1 (p.75).

        Takes no arguments.

        Returns
        -------
        policy_value : numpy.ArrayLike

        """
        policy_value = np.zeros(len(self.board))
        delta = 1
        while delta > self.tol:
            delta = 0
            for state in self.non_terminals:
                v = policy_value[state]
                partial_sum = 0
                for action, action_ix in ACTIONS.items():
                    partial_sum += self.policy[state][
                        action_ix
                    ] * self.get_action_value_function(action, state, policy_value)

                policy_value[state] = partial_sum
                delta = max(delta, abs(v - policy_value[state]))

        return policy_value

    def iterate_policy(self) -> NDArray:
        """
        Iterate policy to minimize loss.

        Note: this will modify the policy attribute (`self.policy`)of the instance. Its
        array will be modified in each iteration until the improvement between states is
        bellow `self.tolerance`. Implemented following Burton-Sutton, section 4.3 (p.80).

        Takes no arguments.

        Returns
        -------
        rv : numpy.ArrayLike
            Policy adopted by an agent trained following the mentioned algorithm. See
            `self.policy` for details on the policy implementation.
        self.policy : numpy.NDArray
            Modified policy.

        """
        iterations = 0
        while True:
            iterations = iterations + 1
            print(f"Iteración : {iterations}")
            rv = self.evaluate_policy()
            print(f"Valor de la política: {rv}")
            policy_stable = True
            for state in self.non_terminals:
                old_action = self.policy[state].copy()
                q_action_state = {
                    act: self.get_action_value_function(act, state, rv)
                    for act in ACTIONS.keys()
                }
                max_actions = [
                    k
                    for k, v in q_action_state.items()
                    if v == max(q_action_state.values())  # use np.allclose() ?
                ]
                for action, action_ix in ACTIONS.items():
                    if action in max_actions:
                        self.policy[state][action_ix] = 1 / len(max_actions)
                    else:
                        self.policy[state][action_ix] = 0
                if not np.allclose(old_action, self.policy[state], atol=self.tol):
                    policy_stable = False

            if policy_stable:
                return self.policy
            else:
                continue

    def board_as_np(self) -> NDArray:
        """Get the board as numpy matrix. Useful to print. Takes no arguments.

        Returns
        -------
        np.array
            Matrix representation of the game board of size (`N`, `N`).

        """
        nodes = np.array(self.board.keys())
        return np.reshape(nodes, (self.board_length, self.board_length))

    def draw_policy(self, highlight_state: Union[int, bool] = False) -> None:
        """
        Display the board using Networkx's default plotting engine.

        Nodes are labeled 1 through `N` and terminal nodes will be colored differently.
        Edges have a width corresponding with the probability value of `self.policy`,
        scaled by a factor of 2 for easier visualization. Only edges with positive
        probability will be drawn.

        Parameters
        ----------
        highlight_state : int, default=False
            State. Only if defined explicitly (only terminal are highlighted otherwise).
            Must be between 0 and `N`².

        Returns no value.

        """
        # define graph and positions for the nodes in the figure
        G = self.graph.copy()

        length = self.board_length
        pos = {node: ((node % length), length - (node // length)) for node in G.nodes}

        # set node's properties and init the figure with them
        labels: dict = {n: n for n in G.nodes}
        colors: List[str] = ['mediumslateblue'] * self.board_size  # default node colors
        for n in self.terminals:
            colors[n] = 'black'
        if highlight_state:
            colors[highlight_state] = 'yellow'

        draw_networkx_nodes(G, pos, node_color=colors, node_shape='d', node_size=299)
        draw_networkx_labels(G, pos, labels=labels)

        # set edges' properties and add to figure
        edges: List[Tuple] = [(s, e) for s, e in G.edges if G[s][e]["weight"] > 0]
        weights: List[float] = [G[s][e]["weight"] * 2 for s, e in edges]

        draw_networkx_edges(
            G, pos, width=weights, edgelist=edges, connectionstyle=f"arc3, rad = {0.19}"
        )

    def interactive_board(
        self,
        filename: str = "./poneme-nombre.html",
        node_scale_factor: int = 10,
    ):
        """Save board as an interactive html file built using pyvis.

        Edge weights (probabilities in `self.policy`) are scaled by `scale factor` for
        easier visualization. A copy of the original weights in string format is seen
        when hovering an edge.

        Parameters
        ----------
        filename : str, default='./plots/poneme-nombre.html'
            Name for the html to save.
        node_scale_factor : int, default=10
            Scaling factor. Must be a positive integer.

        Returns no value.

        """
        G = self.graph.copy()
        pyvis_nt = pyvisnet.Network(directed=True)
        for n in self.terminals:
            pyvis_nt.add_node(n, label=f"{n + 1}", size=13, color="black", shape='star')
        for n in self.non_terminals:
            pyvis_nt.add_node(
                n, label=f"{n + 1}", size=3, color="yellow", shape='circle'
            )
        for start, end in G.edges:
            edge_weight = G.get_edge_data(start, end)["weight"] * node_scale_factor
            if edge_weight > 0:
                pyvis_nt.add_edge(
                    start,
                    end,
                    title=f"{edge_weight / 10}",
                    width=edge_weight,
                    color="blue",
                )
        pyvis_nt.show_buttons()
        pyvis_nt.save_graph(filename)


@click.command(context_settings={'show_default': True})
@click.option("--length", "-l", required=False, default=4, help="Board length.")
@click.option("--gamma", "-g", required=False, default=0.9, help="Discount rate.")
@click.option("--theta", "-t", required=False, default=0.1, help="Tolerance.")
@click.option("--reward", "-r", required=False, default=None, help="Reward.")
@click.option(
    "--policy", "-p", required=False, default='random', help="Initial policy."
)
@click.option("--terminals", required=False, default=None, help="Terminal states.")
def main(length, gamma, theta, reward, policy, terminals):
    print(f"Lado del tabero {length}")
    print(f"Gamma (discount_rate) {gamma}")
    print(f"Theta (Tolerance) {theta}")
    print(f"Politica inicial {policy}")
    juego_1 = Board(
        N=length,
        default_policy=policy,
        tolerance=theta,
        discount_rate=gamma,
        reward=reward,
        terminals=terminals,
    )
    _ = juego_1.iterate_policy()
    juego_1.draw_policy()


if __name__ == "__main__":
    main()  # pylint:disable=E1120
