from combined_feature_model import CombinedFeatureModel
from torch import nn

from evolution_new.evolution_utils import get_small_qubo, remove_hard_constraits_from_qubo, get_edge_data, \
    get_tensor_of_structure
from new_visualisation import qubo_heatmap


class CombinedOneHotModel(CombinedFeatureModel):

    def get_edge_index_and_node_features(self, qubo: list, problem: dict) -> tuple[list[list, list], list]:
        if 'n_colors' in problem:
            calc_qubo = get_small_qubo(qubo, problem['n_colors'])
        elif 'tsp' in problem:
            #calc_qubo = get_small_qubo(qubo, len(problem['cities']))
            calc_qubo = remove_hard_constraits_from_qubo(qubo, problem)
        else:
            calc_qubo = qubo

        qubo_heatmap(calc_qubo)
        edge_index, edge_weights = get_edge_data(calc_qubo)
        node_model, node_features = self.get_node_model_and_features(problem, qubo, calc_qubo)
        print('Node features before: ', node_features)
        node_features = node_model.forward(get_tensor_of_structure(node_features),
                                           get_tensor_of_structure(edge_index).long(),
                                           get_tensor_of_structure(edge_weights)).detach()
        print('Node features after: ', node_features)
        return edge_index, node_features