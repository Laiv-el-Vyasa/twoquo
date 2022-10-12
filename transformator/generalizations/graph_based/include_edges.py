from typing import List, Optional

from networkx import Graph
from numpy.typing import NDArray


def include_edges(
        qubo_in: NDArray,
        graph: Graph,
        positions: Optional[int] = 1,
        score_edges: Optional[float] = None,
        score_edge_weights: Optional[float] = None,
        score_edge_weights_cycles: Optional[float] = None,
        score_non_edges: Optional[float] = None,
        score_non_edges_self: Optional[float] = None,
        score_non_edges_cycles: Optional[float] = None,
) -> None:
    """Encodes the edges of a graph into the input qubo matrix (in place)
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

    :param score_edges: the score for there being an edge between nodes in the
        graph
    :param score_edge_weights: the multiple by which the weights of the edges
        in graph should be scored. For example: if there is an edge between two
        nodes with a weight of 3 and score_edge_weights = 1 then the
        corresponding position in the qubo would have a value of 3. In some
        cases one may wish to scale weights up using a multiple greater than 1
    :param score_edge_weights_cycles: the multiple by which the weights of the
        edges in graph should be scored if the traversal happens on a cycle.
        i.e. the final node back to the start node
    :param score_non_edges: the score for there not being an edge between nodes
        in the graph
    :param score_non_edges_self: the score for a node not having an edge to
        itself
    :param score_non_edges_cycles: the score for nodes which do not have path
        to the node in position 0 to complete a cycle
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

            # SCORE EDGES

            if score_edges:
                for node2 in connected_nodes:

                    # this if statement makes sure that we don't double count
                    # edges otherwise (1,5) and (5,1) would both be scored
                    if node2 >= node1:
                        # CALCULATE SECOND INDEX
                        # same logic as above (see comment)

                        idx2: int = nodes.index(
                            node2
                        ) * positions + position

                        qubo[idx1][idx2] += score_edges

            # SCORE EDGE WEIGHTS

            if score_edge_weights:
                for node2 in connected_nodes:

                    # this if statement makes sure that we don't double count
                    # edges otherwise (1,5) and (5,1) would both be scored
                    # the second part the of the if statement ensures that we
                    # don't score a non edge when the start node is in the last
                    # possible position, i.e. cannot have a descendant
                    if node2 >= node1:
                        if position < positions - 1:
                            # CALCULATE SECOND INDEX
                            # same logic as above (see comment)

                            idx2: int = nodes.index(
                                node2
                            ) * positions + position + 1

                            edge_data = graph.get_edge_data(node1, node2)

                            if weight := edge_data.get('weight'):
                                score = weight * score_edge_weights
                                qubo[idx1][idx2] += score

                        # SCORE EDGE WEIGHTS CYCLES

                        elif position == positions - 1 and score_edge_weights_cycles:  # noqa

                            idx2: int = nodes.index(
                                node2
                            ) * positions

                            edge_data = graph.get_edge_data(node1, node2)

                            if weight := edge_data.get('weight'):
                                score = weight * score_edge_weights_cycles
                                qubo[idx1][idx2] += score

            # SCORE NON EDGES

            if score_non_edges:
                # grab all the nodes to which there is no edge
                unconnected_nodes = [
                    node for node in range(node1 + 1, len(nodes))
                    if node not in connected_nodes
                ]

                for node2 in unconnected_nodes:
                    if positions > 1:
                        # this if statement ensures that we don't score a non
                        # edge when the start node is in the last possible
                        # position, i.e. cannot have a descendant
                        if position < positions - 1:
                            # CALCULATE SECOND INDEX
                            # same logic as above (see comment)

                            idx2: int = nodes.index(
                                node2
                            ) * positions + position + 1

                            qubo[idx1][idx2] += score_non_edges

                        # SCORE NON CYCLE

                        elif position == positions - 1 and score_non_edges_cycles:  # noqa

                            idx2: int = nodes.index(
                                node2
                            ) * positions

                            qubo[idx1][idx2] += score_non_edges_cycles

                    else:  # this is for non-positional problems
                        # in these cases it doesn't make sense to score node1
                        # not having a connection to itself unlike in problems
                        # with more than one position
                        if node1 != node2:
                            # CALCULATE SECOND INDEX
                            # same logic as above (see comment)

                            idx2: int = nodes.index(
                                node2
                            ) * positions + position

                            qubo[idx1][idx2] += score_non_edges

            # SCORE NON SELF-EDGES

            if score_non_edges_self:
                if position < positions - 1:
                    idx2: int = idx1 + 1
                    qubo[idx1][idx2] += score_non_edges_self
