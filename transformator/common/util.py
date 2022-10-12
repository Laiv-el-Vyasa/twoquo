import random
from typing import List, Tuple, Union, Optional

import numpy as np
from networkx import Graph, DiGraph

from networkx.generators.random_graphs import gnm_random_graph
from numpy.random import RandomState
from numpy.typing import NDArray


def gen_graph(
        n_problems: int,
        size: Tuple[int, int],
        seed: Optional[Union[int, RandomState]] = None,
        directed: bool = False,
        weight_range: Optional[Tuple[int, int]] = None
) -> List[Graph]:
    """Generates a list of random networkx graphs

    :param n_problems: number of graphs to be generated
    :param size: tuple of graph size parameters
                 (number of nodes, number of edges)
    :param seed: optional indicator of random generation state
    :param directed: are the graphs directed
    :param weight_range: optional tuple to specify upper and lower bound on
                         randomized edge weights (lower, upper)
    :return: list of randomized Graph objects
    """

    # generate list of random graphs
    graphs: List[Graph] = [
        gnm_random_graph(*size, seed=seed, directed=directed)
        for _ in range(n_problems)
    ]

    # add random weights to edges, if specified
    if weight_range:
        for graph in graphs:
            for (u, v) in graph.edges():
                graph.edges[u, v]['weight'] = random.randint(
                    weight_range[0], weight_range[1]
                )

    return graphs


def gen_subsets_matrix(cls, set_, subsets):
    B = np.zeros((len(set_), len(subsets)))

    for m, x in enumerate(set_):
        for i, subset in enumerate(subsets):
            if x in subset:
                B[m][i] = 1

    return B


def gen_subset_problems(cls, name, cfg, n_problems, size=(20, 25), **kwargs):
    sorting = cfg["problems"][name].get("sorting", False)

    problems = []

    uniques = set()

    set_ = list(range(size[0]))
    for _ in range(n_problems * 3):
        subsets = set()
        for _ in range(size[1]):
            x = list(filter(lambda x: random.random() < 0.5, set_))
            if not x:
                continue
            subsets.add(tuple(x))
        if len(subsets) != size[1]:
            continue
        subsets = sorted(list(subsets))
        if tuple(subsets) in uniques:
            continue
        uniques.add(tuple(subsets))

        B = cls.gen_matrix(set_, subsets)

        # Sort it.
        if sorting:
            y = np.array([2 ** i for i in range(len(subsets))])
            z = B @ y
            idx = np.argsort(z)
            B = B[idx]

        problems.append(B)
        if len(problems) == n_problems:
            break

    # print(name, "generated problems:", len(problems))
    return [{"subset_matrix": matrix} for matrix in problems]


def distance_matrix_to_directed_graph(
        distance_matrix: NDArray
) -> DiGraph:
    """Converts a distance matrix to a weighted directed graph

    :param distance_matrix: a two dimensional matrix with values representing
        edge weights in a directed graph
    :return: a corresponding networkx graph with the weights from the matrix
    """

    # make sure distance matrix is really a matrix and not a list of lists
    distance_matrix: NDArray = np.array(distance_matrix)

    # create a graph with a node for each row
    graph = DiGraph()
    graph.add_nodes_from(range(distance_matrix.shape[0]))

    for x, y in np.ndindex(distance_matrix.shape):  # noqa
        if weight := distance_matrix[x][y]:
            graph.add_edge(x, y)
            graph[x][y]['weight'] = weight

    return graph


def distance_matrix_to_undirected_graph(
        distance_matrix: NDArray
) -> Graph:
    """Converts a distance matrix to a weighted undirected graph

    :param distance_matrix: a two dimensional matrix with values representing
        edge weights in a directed graph
    :return: a corresponding networkx graph with the weights from the matrix
    """

    # make sure distance matrix is really a matrix and not a list of lists
    distance_matrix: NDArray = np.array(distance_matrix)

    # create a graph with a node for each row
    graph = Graph()
    graph.add_nodes_from(range(distance_matrix.shape[0]))

    for x, y in np.ndindex(distance_matrix.shape):  # noqa
        if weight := distance_matrix[x][y]:
            graph.add_edge(x, y)
            graph[x][y]['weight'] = weight

    return graph
