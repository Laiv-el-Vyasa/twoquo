import numpy as np

from transformator.generalizations import include_graph_structure
from transformator.generalizations.graph_based.include_edges import \
    include_edges
from transformator.problems.problem import Problem
from transformator.common.util import gen_graph


class MinimumVertexCover(Problem):
    def __init__(self, cfg, graph, P=8):
        self.graph = graph
        self.P = P

    def gen_qubo_matrix(self):
        n = self.graph.order()
        nodes = list(self.graph.nodes)

        Q = np.zeros((n, n))
        # for i in range(n):
        #     Q[i][i] += 1

        #
        # for edge in self.graph.edges:
        #     idx1 = nodes.index(edge[0])
        #     idx2 = nodes.index(edge[1])
        #     Q[idx1][idx1] -= self.P
        #     Q[idx2][idx2] -= self.P
        #     Q[idx1][idx2] += self.P / 2.
        #     Q[idx2][idx1] += self.P / 2.

        include_graph_structure(
            qubo_in=Q,
            graph=self.graph,
            score_diagonal=1,
            score_nodes_in_edges=-self.P,
        )

        include_edges(
            qubo_in=Q,
            graph=self.graph,
            score_edges=self.P
        )

        Q = (Q + Q.T) / 2

        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size=(20, 25), seed=None,
                     **kwargs):
        graphs = gen_graph(n_problems, size, seed)
        return [{"graph": graph} for graph in graphs]
