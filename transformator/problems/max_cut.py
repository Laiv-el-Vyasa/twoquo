import numpy as np
from tooquo.transformator.common.util import gen_graph
from tooquo.transformator.generalizations import include_graph_structure
from tooquo.transformator.generalizations.graph_based.include_edges import \
    include_edges
from tooquo.transformator.problems.problem import Problem


class MaxCut(Problem):
    def __init__(self, cfg, graph):
        self.graph = graph

    def gen_qubo_matrix(self):
        n = self.graph.order()
        nodes = list(self.graph.nodes)

        # dtype 'b' would be nice here..
        Q = np.zeros((n, n), dtype=np.dtype(np.int32))
        # for edge in self.graph.edges:
        #     idx1 = nodes.index(edge[0])
        #     idx2 = nodes.index(edge[1])
        #     Q[idx1][idx1] += 1
        #     Q[idx2][idx2] += 1
        #     Q[idx1][idx2] -= 1
        #     Q[idx2][idx1] -= 1

        include_graph_structure(
            qubo_in=Q,
            graph=self.graph,
            score_nodes_in_edges=-1,
        )

        include_edges(
            qubo_in=Q,
            graph=self.graph,
            score_edges=2
        )

        Q = (Q + Q.T)/2
        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size, seed=None, **kwargs):
        graphs = gen_graph(n_problems, size, seed)
        return [{"graph": graph} for graph in graphs]


class MaxCutMemoryEfficient(Problem):
    def __init__(self, cfg, edge_list, n):
        self.edge_list = edge_list
        self.n = n

    def gen_qubo_matrix(self):
        Q = np.zeros((self.n, self.n), dtype=np.dtype('b'))
        for edge in self.edge_list:
            idx1 = edge[0]
            idx2 = edge[1]
            Q[idx1][idx1] += 1
            Q[idx2][idx2] += 1
            Q[idx1][idx2] -= 1
            Q[idx2][idx1] -= 1

        return -Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size, seed=None, **kwargs):
        data = []
        for i in range(n_problems):
            if i % 100000 == 0:
                print(i)
            data.append(list(gen_graph(1, size, seed)[0].edges))
        return [{"edge_list": graph, "n": size[0]} for graph in data]
