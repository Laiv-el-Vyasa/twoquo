from combined_model import CombinedModel
from torch import nn

from evolution_new.evolution_utils import get_diagonal_of_qubo, get_min_of_tsp_qubo_line_normalized


class CombinedFeatureModel(CombinedModel):

    def get_node_model_and_features(self, problem: dict, qubo: list, calc_qubo) -> tuple[nn.Module, list]:
        return_node_model = self.node_model
        return_node_features = get_diagonal_of_qubo(calc_qubo)
        if 'tsp' in problem:
            return_node_features = get_min_of_tsp_qubo_line_normalized(qubo, len(problem['cities']))
            return_node_model = self.node_model_normalized
        return return_node_model, return_node_features
