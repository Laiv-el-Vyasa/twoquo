import copy
import itertools
import numpy as np

from transformator.generalizations.graph_based.include_graph_structure import include_graph_structure
from transformator.generalizations.graph_based.include_edges import \
    include_edges
from transformator.problems.max_cut import generate_random_edge_number
from transformator.problems.problem import Problem
from transformator.common.util import gen_graph


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
        adapted_size = copy.deepcopy(size)
        for i in range(n_problems):
            if cfg['problems']['GC']['random_edges']:
                adapted_size[1] = generate_random_edge_number(size)
            graphs.extend(gen_graph(1, adapted_size, seed))
        return [
            {"graph": graph, "n_colors": n_colors}
            for graph in graphs
        ]
