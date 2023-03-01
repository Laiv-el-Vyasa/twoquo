import numpy as np
from pygad import pygad
from pygad import torchga
import pygad.torchga
import torch

from multiprocessing import Pool

from evolution_utils import get_edge_data, get_diagonal_of_qubo, get_tensor_of_structure, apply_approximation_to_qubo, \
    linearize_qubo
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

    def get_approximation(self, qubo_list: list, problem_list: list) -> list:
        approxed_qubo_list = [None for _ in qubo_list]
        with Pool() as pool:
            argument_list = [qubo_list, problem_list, [i for i in range(len(qubo_list))]]
            for approxed_qubo, index in pool.starmap(self.get_approxed_qubo, argument_list):
                approxed_qubo_list[index] = approxed_qubo
        return approxed_qubo_list

    def get_approxed_qubo(self, qubo, problem, index):
        edge_index, edge_weight = get_edge_data(qubo)
        node_model, node_features = self.get_node_model_and_features(problem, qubo)
        node_features = node_model.forward(get_tensor_of_structure(node_features),
                                           get_tensor_of_structure(edge_index).long(),
                                           get_tensor_of_structure(edge_weight)).detach()
        approx_mask = np.ones((len(qubo), len(qubo)))
        node_mean_tensor_list = []
        for edge_0, edge_1 in zip(edge_index[0], edge_index[1]):
            node_features_0 = np.array(node_features[edge_0].numpy())
            node_features_1 = np.array(node_features[edge_1].numpy())
            node_mean_tensor_list.append(np.mean([node_features_0, node_features_1], axis=0))

        edge_descision_list = self.edge_model.forward(get_tensor_of_structure(node_mean_tensor_list)).detach()
        for idx, edge_descision in enumerate(edge_descision_list):
            if edge_descision.detach() <= 0:
                approx_mask[edge_index[0][idx]][edge_index[1][idx]] = 0
        return np.multiply(qubo, approx_mask), index

    def get_node_model_and_features(self, problem, qubo):
        return_node_model = self.node_model
        return_node_features = get_diagonal_of_qubo(qubo)
        if 'graph' in problem and 'tsp' in problem:
            # print('Normalized model choosen!')
            return_node_model = self.node_model_normalized
        # return_node_features = get_mean_of_qubo_line(qubo)
        return return_node_model, return_node_features

    def set_model_weights_from_pygad(self, pygad_chromosome: list):
        node_weights, edge_weights = self.get_model_weight_dicts(pygad_chromosome)
        self.apply_weights_to_models(node_weights, edge_weights)

    def get_initial_population(self, population_size: int) -> list:
        torch_ga_node = pygad.torchga.TorchGA(model=self.node_model, num_solutions=population_size)
        torch_ga_edge = pygad.torchga.TorchGA(model=self.edge_model, num_solutions=population_size)
        return np.append(torch_ga_node.population_weights, torch_ga_edge.population_weights, axis=-1)

    def save_best_model(self, pygad_chromosome: list, model_name: str):
        node_weights, edge_weights = self.get_model_weight_dicts(pygad_chromosome)
        torch.save(node_weights, f'best_models/{model_name}_node')
        torch.save(edge_weights, f'best_models/{model_name}_edge')

    def load_best_model(self, model_name: str):
        try:
            node_weights = torch.load(f'best_models/{model_name}_node')
            edge_weights = torch.load(f'best_models/{model_name}_edge')
            self.apply_weights_to_models(node_weights, edge_weights)
        except FileNotFoundError:
            print('Model hasn`t been trained yet!')

    def get_model_weight_dicts(self, pygad_chromosome: list) -> tuple[dict, dict]:
        model_weights_dict_node = torchga.model_weights_as_dict(model=self.node_model,
                                                                weights_vector=pygad_chromosome[:self.node_edge_cutoff])
        model_weights_dict_edge = torchga.model_weights_as_dict(model=self.edge_model,
                                                                weights_vector=pygad_chromosome[self.node_edge_cutoff:])
        return model_weights_dict_node, model_weights_dict_edge

    def apply_weights_to_models(self, node_weights: dict, edge_weights: dict):
        self.node_model.load_state_dict(node_weights)
        self.node_model_normalized.load_state_dict(node_weights)
        self.edge_model.load_state_dict(edge_weights)