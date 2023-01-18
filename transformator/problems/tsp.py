import copy
from typing import Tuple, List

import numpy as np
from networkx import Graph
from numpy.typing import NDArray

from transformator.common.exceptions import BadProblemParametersError, \
    EmptyGraphError
from transformator.common.util import gen_graph
from transformator.generalizations.graph_based.include_graph_structure import include_graph_structure
from transformator.generalizations.graph_based.include_edges import \
    include_edges
from transformator.problems.max_cut import generate_random_edge_number
from transformator.problems.problem import Problem


class TSP(Problem):
    def __init__(
            self,
            graph: Graph
    ):
        # raise exception if input graph is empty
        if not graph.nodes:
            raise BadProblemParametersError from EmptyGraphError(
                'Graph consists of 0 nodes.'
            )
        self.graph: Graph = graph

    def gen_qubo_matrix(
            self
    ) -> NDArray:
        n: int = self.graph.order()

        # select a scaling constant — similarly, a is the weighting for H_A
        # while b is the scaling constant for H_B, the method for deciding the
        # value of a and b is located in Lucas 2014 in the TSP section.
        b: int = 1
        edge_weights = [
            self.graph.get_edge_data(edge[0], edge[1]).get('weight')
            for edge in self.graph.edges
        ]
        a: int = b * max(edge_weights, default=0) + 1

        # init Q matrix with dimensions n * n
        Q: NDArray = np.zeros((n * n, n * n))

        include_graph_structure(
            qubo_in=Q,
            graph=self.graph,
            positions=n,
            score_diagonal=-2*a,
            score_one_node_many_positions=2*a,
            score_many_nodes_one_position=2*a,
        )

        include_edges(
            qubo_in=Q,
            graph=self.graph,
            positions=n,
            score_non_edges=a,
            score_non_edges_cycles=a,
            score_edge_weights=b,
            score_edge_weights_cycles=b,
        )

        Q = (Q + Q.T) / 2  # make it symmetrical
        return Q

    @classmethod
    def gen_problems(
            self,
            cfg,
            n_problems,
            size: Tuple[int, int] = (10, 10),
            #size,
            weight_range=(1, 10),
            **kwargs
    ):
        print(size)
        #graphs = []
        #adapted_size = copy.deepcopy(size)
        #for i in range(n_problems):
        #    if cfg['problems']['TSP']['random_edges']:
        #        adapted_size[1] = generate_random_edge_number(size)
        #    graphs.extend(gen_graph(1, adapted_size, weight_range=weight_range))
        graphs: List[Graph] = gen_graph(
            n_problems=n_problems, size=size, weight_range=weight_range
        )
        problems = []
        for graph in graphs:
            problems.append({'graph': graph})
        return problems
