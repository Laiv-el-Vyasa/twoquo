import copy

import numpy as np
from transformator.common.util import gen_graph
from transformator.generalizations.graph_based.include_graph_structure import include_graph_structure
from transformator.generalizations.graph_based.include_edges import \
    include_edges
from transformator.problems.problem import Problem
from numpy import random


def get_random_node_number(size):
    rng = random.default_rng()
    return rng.integers(size[0], size[1])


def get_random_edge_number(cfg, nodes):
    rng = random.default_rng()
    edges = 0
    while edges < 1 or edges > ((nodes * (nodes + 1)) / 2):
        edges = rng.normal(nodes * cfg['problems']['MC']['edge_node_factor'], nodes)
    return int(np.round(edges))


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
    def gen_problems(self, cfg, n_problems, size=(16, 64), seed=None, **kwargs):
        graphs = []
        for i in range(n_problems):
            nodes = get_random_node_number(size)
            edges = get_random_edge_number(cfg, nodes)
            graphs.extend(gen_graph(1, (nodes, edges), seed))
        return [{"graph": graph} for graph in graphs]

