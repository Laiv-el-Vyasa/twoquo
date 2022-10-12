import random
from typing import List, Tuple
import itertools
import numpy as np
from numpy.typing import NDArray
from networkx import Graph

from transformator.common.exceptions import BadProblemParametersError
from transformator.common.exceptions import EmptyGraphError
from transformator.generalizations import include_graph_structure
from transformator.generalizations.graph_based.include_edges import \
    include_edges
from transformator.problems.problem import Problem
from transformator.common.util import gen_graph


class LongestPath(Problem):
    def __init__(
            self,
            graph: Graph,
            start_node,
            terminal_node,
            steps: int
    ):
        self.graph: Graph = graph
        self.start_node = start_node
        self.terminal_node = terminal_node
        self.steps = steps
        self.k = steps + 1

    def gen_qubo_matrix(
            self
    ) -> NDArray:
        n: int = self.graph.order()

        # test for bad parameters
        if self.start_node == self.terminal_node:
            raise BadProblemParametersError(
                'Start node is equal to terminal node.'
            )
        if self.start_node not in range(0, self.graph.order()):
            raise BadProblemParametersError(
                f'Start node index is out of bounds.\n'
                f'start_node: {self.start_node} but '
                f'graph.order(): {self.graph.order()}.'
            )
        if self.terminal_node not in range(0, self.graph.order()):
            raise BadProblemParametersError(
                f'Terminal node index is out of bounds.\n'
                f'terminal_node: {self.terminal_node} but '
                f'graph.order(): {self.graph.order()}.'
            )
        if self.steps > self.graph.order() - 1:
            raise BadProblemParametersError(
                f'Number of steps is out of bounds.\n'
                f'The value of steps should be between 0 and graph.order()-1.'
                f'steps: {self.steps} but graph.order(): {self.graph.order()}.'
            )
        for (u, v) in self.graph.edges:
            if self.graph.get_edge_data(u, v).get('weight') <= 0:
                raise BadProblemParametersError(
                    f'Negative weight value.'
                    f'Weight ({u}, {v}): '
                    f'{self.graph.get_edge_data(u, v).get("weight")}.'
                )
        if not self.graph.nodes:
            raise BadProblemParametersError from EmptyGraphError(
                'graph consists of 0 nodes.'
            )

        # init Q matrix with size n * steps to allow for
        # vertex, position selection
        Q = np.zeros((n * self.k, n * self.k))

        # calculate an appropriate scaling constant to be applied to penalties
        # ensures an impossible path is never selected, even if it provides the
        # maximum edge weight
        # background on how penalty_scale is chosen on page 7 of the
        # paper in test_longest_path.py
        edge_weights = [
            self.graph.get_edge_data(edge[0], edge[1]).get('weight')
            for edge in self.graph.edges
        ]
        penalty_scale: int = self.steps * max(edge_weights, default=0)

        # start penalty
        # penalizes vectors which do not start on the correct vertex

        if not self.graph.has_node(self.start_node):
            raise Exception(
                f'Error: {self.start_node} not a valid node in graph'
            )
        else:
            index: int = self.k * self.start_node + 0
            # + 0 because 0 is the starting position
            Q[index][index] += -1 * penalty_scale

        # terminal penalty
        # penalizes vectors which do not end on the correct vertex

        if not self.graph.has_node(self.terminal_node):
            raise Exception(
                f'Error: {self.terminal_node} not a valid node in graph'
            )
        else:
            index: int = self.k * self.terminal_node + self.steps
            # + self.steps because self.steps is the terminal position
            Q[index][index] += -1 * penalty_scale

        # # one-vertex-per-position penalty
        # # penalizes having multiple nodes in the same position
        #
        # for position in range(self.k):
        #     for node in range(n):
        #         index: int = self.k * node + position
        #
        #         # reward the existence of a node in that position
        #         Q[index][index] += -1 * penalty_scale
        #
        #         # penalize multiple nodes in that position
        #         for node_next in range(node + 1, n):
        #             index_next: int = self.k * node_next + position
        #
        #             Q[index][index_next] += 2 * penalty_scale

        # # one-visit-per-vertex penalty
        #
        # for node, position in itertools.product(range(n), range(self.k)):
        #     index: int = self.k * node + position
        #
        #     # one-visit-per-vertex penalty
        #     # penalizes having one node be in multiple positions
        #
        #     for position_next in range(position + 1, self.k):
        #         index_next = self.k * node + position_next
        #         Q[index][index_next] += 1 * penalty_scale

        # for node, position in itertools.product(range(n), range(self.steps)):
        #     index: int = self.k * node + position
        #
        #     # edge weight
        #     # objective term: rewards traversing edges based on their weights
        #
        #     for node_next in range(node, n):
        #         index_next: int = self.k * node_next + position + 1
        #
        #         # reward if edge exists
        #         # otherwise penalize impossible traversal
        #         if edge_data := self.graph.get_edge_data(node, node_next):
        #             weight: int = -1 * edge_data.get('weight')
        #         else:
        #             weight: int = 1 * penalty_scale
        #
        #         Q[index][index_next] += weight

        include_graph_structure(
            qubo_in=Q,
            graph=self.graph,
            positions=self.k,
            score_diagonal=-penalty_scale,
            score_many_nodes_one_position=2*penalty_scale,
            score_one_node_many_positions=penalty_scale,
        )

        include_edges(
            qubo_in=Q,
            graph=self.graph,
            positions=self.k,
            score_edge_weights=-1,
            score_non_edges=penalty_scale,
            score_non_edges_self=penalty_scale
        )

        # make it symmetric
        Q = (Q + Q.T) / 2


        return Q

    @classmethod
    def gen_problems(
            cls,
            cfg,
            n_problems: int,
            size: Tuple[int, int] = (10, 10),
            weight_range=(1, 10),
            **kwargs
    ):
        graphs: List[Graph] = gen_graph(
            n_problems=n_problems, size=size, weight_range=weight_range
        )
        problems = []
        for graph in graphs:
            # select two random nodes for start/terminus
            random_nodes: List = random.sample(list(graph.nodes()), 2)
            problems.append({
                "graph": graph,
                "start_node": random_nodes[0],
                "terminal_node": random_nodes[1],
                # select a number of steps <= number of edges
                "steps": random.sample(range(1, size[1]), 1)[0],
            })
        return problems
