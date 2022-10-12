import numpy as np

from transformator.generalizations.graph_based.include_graph_structure import include_graph_structure
from transformator.generalizations.graph_based.include_edges import \
    include_edges
from transformator.problems.problem import Problem
from transformator.common.util import gen_graph


class MaxClique(Problem):
    def __init__(self, cfg, graph):
        self.graph = graph

    def gen_qubo_matrix(self):
        n = self.graph.order()

        Q = np.zeros((n, n))

        # for i in range(n):
        #     for j in range(n):
        #         if i == j:
        #             Q[i][i] = -1
        #         if i < j and (i, j) not in self.graph.edges:
        #             Q[i][j] = 2
        #             Q[j][i] = 2

        include_graph_structure(
            qubo_in=Q,
            graph=self.graph,
            score_diagonal=-1,
        )

        include_edges(
            qubo_in=Q,
            graph=self.graph,
            score_non_edges=2 * 2
        )

        Q = (Q + Q.T) / 2

        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size, seed=None, **kwargs):
        graphs = gen_graph(n_problems, size, seed)
        return [{"graph": graph} for graph in graphs]
