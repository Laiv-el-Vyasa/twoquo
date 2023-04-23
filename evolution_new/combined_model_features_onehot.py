import numpy as np

from combined_feature_model import CombinedFeatureModel
from torch import nn

from evolution_new.evolution_utils import get_small_qubo, remove_hard_constraits_from_qubo, get_edge_data, \
    get_tensor_of_structure, get_diagonal_of_qubo, get_min_of_tsp_qubo_line_normalized, \
    get_min_of_tsp_qubo_line_normalized_onehot, get_reducability_number_onehot, get_number_of_edges_for_gc
from new_visualisation import qubo_heatmap


class CombinedOneHotFeatureModel(CombinedFeatureModel):

    def get_edge_index_and_node_features(self, qubo: list, problem: dict) -> tuple[list[list, list], list]:
        if 'tsp' in problem or 'n_colors' in problem:
            #calc_qubo = get_small_qubo(qubo, len(problem['cities']))
            calc_qubo = remove_hard_constraits_from_qubo(qubo, problem, False)
        else:
            calc_qubo = qubo

        # qubo_heatmap(calc_qubo)
        edge_index, edge_weights = get_edge_data(calc_qubo)
        node_model, node_features = self.get_node_model_and_features(problem, qubo, calc_qubo)
        # print('Node features before: ', node_features)
        node_features = node_model.forward(get_tensor_of_structure(node_features),
                                           get_tensor_of_structure(edge_index).long(),
                                           get_tensor_of_structure(edge_weights)).detach()
        # print('Node features after: ', node_features)
        return edge_index, node_features

    def get_node_model_and_features(self, problem: dict, qubo: list, calc_qubo) -> tuple[nn.Module, list]:
        return_node_model = self.node_model
        return_node_features = get_diagonal_of_qubo(calc_qubo)
        if 'tsp' in problem:
            return_node_features = get_min_of_tsp_qubo_line_normalized_onehot(qubo, len(problem['cities']))
            # return_node_model = self.node_model_normalized
        elif 'n_colors' in problem:
            return_node_features = get_number_of_edges_for_gc(qubo, problem['n_colors'])
        return return_node_model, return_node_features

    def get_approx_mask(self, edge_index: list[list, list], node_mean_tensor_list: list,
                        qubo: list, problem: dict) -> list:
        approx_mask = np.ones((len(qubo), len(qubo)))
        edge_decision_list = self.edge_model.forward(get_tensor_of_structure(node_mean_tensor_list)).detach()
        for idx, edge_decision in enumerate(edge_decision_list):
            if edge_decision.detach() <= 0:
                if not edge_index[0][idx] == edge_index[1][idx]:
                    approx_mask[edge_index[0][idx]][edge_index[1][idx]] = 0
        # qubo_heatmap(qubo)
        # qubo_heatmap(approx_mask)
        return approx_mask