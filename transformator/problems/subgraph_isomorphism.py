import itertools
import math

import numpy as np
from networkx import erdos_renyi_graph

from evolution_new.new_visualisation import qubo_heatmap
from transformator.problems.problem import Problem
from transformator.common.util import gen_graph
from numpy import random


def get_random_node_number(size: list[int, int]) -> int:
    rng = random.default_rng()
    return rng.integers(size[0], size[1] + 1)


# Calculate critical value for phase transition
def get_critical_value(n1: int, n2: int, e2: float) -> float:
    # <Sol> = t_ * e2 ^ (e1 * (n1 over 2)) -> <Sol> = 1
    # e1 = log_e2(1 / t_) * (1 / (n1 over 2)
    t_ = get_possible_assignments(n1, n2)
    n1_2 = math.comb(n1, 2)
    e1_critival = (np.log(1 / t_) / np.log(e2)) * (1 / n1_2)
    if e1_critival > 1:  # Fallback when critical value to big
        e1_critival = 1.
    return e1_critival


def choose_edge_probability(n1: int, n2: int, e2: float) -> float:
    e1_critical = get_critical_value(n1, n2, e2)
    e1 = 0
    rng = random.default_rng()
    # print('Critical value: ', e1_critical)
    while not 0 < e1 < 1:
        e1 = rng.normal(e1_critical, 0.2)
    # print('Edge probability: ', e1)
    return e1


def get_possible_assignments(n1: int, n2: int) -> int:
    t_ = 1
    t = n2
    while t > n1:
        t_ *= t
        t -= 1
    return t_


class SubGraphIsomorphism(Problem):
    """Generalization of Graph Isomorphism."""
    def __init__(self, cfg, graph1, graph2, a=1, b=2):
        self.graph1 = graph1
        self.graph2 = graph2
        self.a = a
        self.b = b

    def gen_qubo_matrix(self):
        n1 = self.graph1.order()
        n2 = self.graph2.order()
        Q = np.zeros((n1 * n2, n1 * n2))

        for i in range(n1):
            for j in range(n2):
                idx = i * n2 + j
                Q[idx][idx] -= 1 * self.a
            for k, m in itertools.combinations(list(range(n2)), 2):
                idx1 = i * n2 + k
                idx2 = i * n2 + m
                Q[idx1][idx2] += 2 * self.a
                Q[idx2][idx1] += 2 * self.a

        for j in range(n2):
            for i in range(n1):
                idx = i * n2 + j
                Q[idx][idx] -= 1
            for k, m in itertools.combinations(list(range(n1)), 2):
                idx1 = k * n2 + j
                idx2 = m * n2 + j
                Q[idx1][idx2] += 2
                Q[idx2][idx1] += 2

        nodes = list(self.graph1.nodes)
        for edge in self.graph1.edges:
            i = nodes.index(edge[0])
            j = nodes.index(edge[1])

            for i_dash in range(n2):
                idx1 = i * n2 + i_dash
                for j_dash in range(n2):
                    idx2 = j * n2 + j_dash
                    if (i_dash, j_dash) in self.graph2.edges:
                        continue
                    Q[idx1][idx2] += 1 * self.b
                    Q[idx2][idx1] += 1 * self.b
        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size1, size2, seed=None, **kwargs):
        graphs1 = []  # Pattern graph
        graphs2 = []  # Target graph
        # NOTE: size1 should be smaller than size2 or equal to it.
        for i in range(n_problems):
            n1 = get_random_node_number(size1)
            n2 = get_random_node_number(size2)
            if n1 > n2:
                n1, n2 = n2, n1
            e2 = cfg["problems"]["SGI"].get("edge_p2", 0.4)
            e1 = choose_edge_probability(n1, n2, e2)
            graphs1.append(erdos_renyi_graph(n1, e1))
            graphs2.append(erdos_renyi_graph(n2, e2))


        #graphs1 = gen_graph(n_problems, size1, seed)
        #print('G1 nodes: ', graphs1[0].nodes)
        #print('G1 edges: ', graphs1[0].edges)
        #graphs2 = gen_graph(n_problems, size2, seed)
        #print('G2 nodes: ', graphs2[0].nodes)
        #print('G2 edges: ', graphs2[0].edges)
        return [
            {"graph1": graph1, "graph2": graph2}
            for graph1, graph2 in zip(graphs1, graphs2)
        ]
