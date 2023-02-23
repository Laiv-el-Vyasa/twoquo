import numpy as np
from pygad import pygad
from pygad import torchga
import pygad.torchga


from learning_model import LearningModel
from neural_networks import CombinedNodeFeaturesUwu, CombinedEdgeDecisionUwu

combined_model_list = {
    'combinedModelUwU': [
        CombinedNodeFeaturesUwu,
        CombinedEdgeDecisionUwu
    ]
}


class CombinedModel(LearningModel):
    def __init__(self, network_information: dict):
        network_name = network_information['network_name']
        node_features = network_information['node_features']
        self.node_model = combined_model_list[network_name][0](node_features, False)
        self.node_model_normalized = combined_model_list[network_name][0](node_features, True)
        self.edge_model = combined_model_list[network_name][1](node_features)
        # Get the cutoff point, where the pygad string has to be sliced
        self.node_edge_cutoff = sum(p.numel() for p in self.node_model.parameters() if p.requires_grad)

    def get_approximation(self, qubo_list, problem_list):
        pass

    def set_model_weights_from_pygad(self, pygad_chromosome):
        model_weights_dict_node = torchga.model_weights_as_dict(model=self.node_model,
                                                                weights_vector=pygad_chromosome[:self.node_edge_cutoff])
        self.node_model.load_state_dict(model_weights_dict_node)
        self.node_model_normalized.load_state_dict(model_weights_dict_node)

        model_weights_dict_edge = torchga.model_weights_as_dict(model=self.edge_model,
                                                                weights_vector=pygad_chromosome[self.node_edge_cutoff:])
        self.edge_model.load_state_dict(model_weights_dict_edge)

    def get_initial_population(self, population_size):
        torch_ga_node = pygad.torchga.TorchGA(model=self.node_model, num_solutions=population_size)
        torch_ga_edge = pygad.torchga.TorchGA(model=self.edge_model, num_solutions=population_size)
        return np.append(torch_ga_node.population_weights, torch_ga_edge.population_weights, axis=-1)

