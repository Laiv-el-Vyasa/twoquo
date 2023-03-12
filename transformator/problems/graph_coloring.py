import copy
import itertools

import numpy as np
from numpy import random
from networkx import erdos_renyi_graph

from transformator.generalizations.graph_based.include_graph_structure import include_graph_structure
from transformator.generalizations.graph_based.include_edges import \
    include_edges
from transformator.problems.problem import Problem
from transformator.common.util import gen_graph

critical_connectivities = {
    3: 4.69,
    4: 8.27,
    5: 13.69
}


def get_random_coloring() -> int:
    rng = random.default_rng()
    return rng.integers(3, 5)


def get_random_node_number(size: tuple[int, int]) -> int:
    rng = random.default_rng()
    return rng.integers(size[0], size[1])


def get_random_edge_probability(nodes: int, n_colors: int) -> float:
    rng = random.default_rng()
    critical_connectivity = critical_connectivities[n_colors]
    connectivity = 0
    while 0 < connectivity < nodes:
        connectivity = rng.normal(critical_connectivity, critical_connectivity / 2)
    return connectivity / (nodes - 1)


class GraphColoring(Problem):
    def __init__(self, cfg, graph, n_colors, P=4):
        self.graph = graph
        self.n_colors = n_colors
        self.P = P

    def gen_qubo_matrix(self):
        n = self.graph.order() * self.n_colors
        nodes = list(self.graph.nodes)

        Q = np.zeros((n, n))

        # for i in range(n):
        #     Q[i][i] -= self.P

        # for i, x in enumerate(nodes):
        #     cols = [i * self.n_colors + c for c in range(self.n_colors)]
        #     tuples = itertools.combinations(cols, 2)
        #     for j, k in tuples:
        #         Q[j][k] += self.P
        #         Q[k][j] += self.P

        # for edge in self.graph.edges:
        #     idx1 = nodes.index(edge[0])
        #     idx2 = nodes.index(edge[1])
        #     for c in range(self.n_colors):
        #         idx1c = idx1 * self.n_colors + c
        #         idx2c = idx2 * self.n_colors + c
        #         Q[idx1c][idx2c] += self.P / 2.
        #         Q[idx2c][idx1c] += self.P / 2.

        include_graph_structure(
            qubo_in=Q,
            graph=self.graph,
            positions=self.n_colors,
            score_diagonal=-self.P,
            score_one_node_many_positions=self.P * 2,  # TODO: see if this is obvious
        )

        include_edges(
            qubo_in=Q,
            graph=self.graph,
            positions=self.n_colors,
            score_edges=self.P,
        )

        Q = (Q + Q.T) / 2

        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size, n_colors, seed=None, **kwargs):
        graphs = []
        n_colors_list = []
        for i in range(n_problems):
            n_colors = get_random_coloring()
            n_colors_list.append(n_colors)
            nodes = get_random_node_number(size)
            graphs.extend(erdos_renyi_graph(nodes, get_random_edge_probability(nodes, n_colors)))
        return [
            {"graph": graph, "n_colors": n_colors}
            for (graph, n_colors) in zip(graphs, n_colors_list)
        ]
