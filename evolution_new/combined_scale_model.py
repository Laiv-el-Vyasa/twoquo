import numpy as np

from combined_model import CombinedModel

# Special class used to train a model, capable of approximating roughly a given percentage of entries (scale parameter)


class CombinedScaleModel(CombinedModel):
    # Method overridden from parent class, calling the new method to get the approximated QUBO
    def get_approximation(self, problem_dict: dict) -> dict:
        problem_list, qubo_list, scale_list = problem_dict['problem_list'], problem_dict['qubo_list'], \
                                              problem_dict['scale_list']
        # print(problem_list)
        approxed_qubo_list = []

        for index, (qubo, problem, scale) in enumerate(zip(qubo_list, problem_list, scale_list)):
            approxed_qubo, _ = self.get_scaled_approxed_qubo(qubo, problem, scale, index)
            approxed_qubo_list.append(approxed_qubo)
        problem_dict['approxed_qubo_list'] = approxed_qubo_list
        return problem_dict

    # Method mirroring the approximation method from th parent class
    # Adding the scale parameter to the node-features, so the edge-decision model can include its value
    def get_scaled_approxed_qubo(self, qubo: list, problem: dict, scale: float, index: int) -> tuple[list, int]:
        edge_index, node_features = self.get_edge_index_and_node_features(qubo, problem)
        # print('Scale: ', scale)
        # print(f'Problem {index}, node_features: {node_features}')
        node_mean_tensor_list = []
        for edge_0, edge_1 in zip(edge_index[0], edge_index[1]):
            node_features_0 = np.array(node_features[edge_0].numpy())
            node_features_1 = np.array(node_features[edge_1].numpy())
            node_mean_tensor_list.append(np.concatenate([np.mean([node_features_0, node_features_1], axis=0),
                                                         np.array([scale])]))

        approx_mask = self.get_approx_mask(edge_index, node_mean_tensor_list, qubo, problem)
        return np.multiply(qubo, approx_mask), index

