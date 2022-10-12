from typing import List, Optional, Union

from networkx import Graph, DiGraph
from numpy.typing import NDArray


def include_graph_structure(
        qubo_in: NDArray,
        graph: Union[Graph, DiGraph],
        positions: Optional[int] = 1,
        score_diagonal: Optional[float] = None,
        score_nodes_in_edges: Optional[float] = None,
        score_one_node_many_positions: Optional[float] = None,
        score_many_nodes_one_position: Optional[float] = None,
) -> None:
    """Encodes the structure of a graph into the input qubo matrix (in place)
    in order to generalize graph based problem transformations

    ABOUT SCORE_X PARAMETERS
    Different problems require penalties or rewards to be placed in different
    ares of the qubo matrix. "score_x" parameters are unsigned, meaning your
    implementation can decide to whether the given positions in the qubo matrix
    should be assigned rewards (negative input values) or penalties (positive
    input values).

    :param qubo_in: the qubo on which the graph structure is encoded (in place)
    :param graph: the graph from the problem
    :param positions: the number of positions in the problem, for example n
        colors in graph coloring, or positions in longest path

    :param score_diagonal: the score the diagonal of the qubo
    :param score_nodes_in_edges: the score along the diagonal for nodes which
        have edges connected to them
    :param score_one_node_many_positions: the score for a single node being
        present in more than position
    :param score_many_nodes_one_position: the score for multiple nodes being
        present in a single position
    """

    # copy input qubo
    qubo: NDArray = qubo_in

    # get nodes and their cartesian product
    nodes = list(graph.nodes)

    for node1 in nodes:
        # get the edges of the node1 in question
        connected_nodes: List = [node for _, node in graph.edges(node1)]

        for position in range(positions):

            # CALCULATE INDEX
            # simple case:     the positions loop is executed once and position
            #                  has the value 0 so the below formula still
            #                  returns the correct index for each node
            # positional case: the below formula calculates the correct index
            #                  for each node and each position

            # note: we use nodes.index(_) so we don't need to worry about the
            #       names of the nodes, i.e. a graph with a node 0 and a graph
            #       that starts with node 1 will be treated the same

            idx1: int = nodes.index(node1) * positions + position

            # SCORE DIAGONAL

            if score_diagonal:
                qubo[idx1][idx1] += score_diagonal

            # SCORE NODES IN EDGES

            if score_nodes_in_edges:
                for _ in connected_nodes:
                    qubo[idx1][idx1] += score_nodes_in_edges

            # SCORE ONE NODE TO MANY POSITIONS

            if score_one_node_many_positions:
                other_positions = range(position + 1, positions)
                for position2 in other_positions:
                    # CALCULATE SECOND INDEX
                    # same logic as above (see comment)

                    idx2: int = nodes.index(
                        node1
                    ) * positions + position2

                    qubo[idx1][idx2] += score_one_node_many_positions

            # SCORE MANY NODES TO ONE POSITION

            if score_many_nodes_one_position:
                other_nodes = range(node1 + 1, len(nodes))
                for node2 in other_nodes:
                    # CALCULATE SECOND INDEX
                    # same logic as above (see comment)

                    idx2: int = nodes.index(
                        node2
                    ) * positions + position

                    qubo[idx1][idx2] += score_many_nodes_one_position
