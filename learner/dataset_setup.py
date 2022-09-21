import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.stats as st
from visualisation import approx_quality_graphs_problems_ml

from recommendation import RecommendationEngine


def get_optimal_approx_percent(approx_dict, approx_steps, solver, lower_bound):
    optimization_array = np.ones(approx_steps)
    for i in range(approx_steps):
        step = i + 1
        quality = approx_dict[str(step)][solver]
        percent = step / (approx_steps + 1)
        if quality > lower_bound:
            optimization_array[i] = (1 - quality) + (1 - percent)
    return (np.max(np.argmin(optimization_array)) + 1) / approx_steps


class DatabaseSetup:
    def __init__(self, problem_number, solver):
        self.db = RecommendationEngine().get_database()
        self.problem_number = problem_number
        self.solver = solver

    def get_data_for_simple_learning(self, lower_bound):
        X_classes = []
        Y_target = []
        for _, metadata in self.db.iter_metadata():
            # print(metadata)
            problem_one_hot = np.zeros(self.problem_number)
            problem_one_hot[metadata.problem] = 1
            X_classes.append(problem_one_hot)
            target_percent = np.zeros(1)
            target_percent[0] = get_optimal_approx_percent(metadata.approx_solution_quality, metadata.approx, self.solver, lower_bound)
            Y_target.append(target_percent)
            #Y_target.append(
            #    get_optimal_approx_percent(metadata.approx_solution_quality, metadata.approx, solver, lower_bound))
        #print(np.array(Y_target))
        return np.array(X_classes), np.array(Y_target)

    def aggregate_saved_problem_data(self, solver):
        global approx_steps
        approx_strategy = False
        aggregation_array = []
        for i, (_, metadata) in enumerate(self.db.iter_metadata()):
            approx_steps = metadata.approx
            if i == 0:
                aggregation_array = self.prepare_aggregation_array(approx_steps)
                approx_strategy = metadata.approx_strategy
            for step in range(approx_steps + 1):
                if step == 0:
                    quality = 1
                else:
                    quality = metadata.approx_solution_quality[str(step)][solver]
                aggregation_array[metadata.problem][step].append(quality)
        return np.mean(aggregation_array, axis=2), [n / (approx_steps + 1) for n in range(approx_steps + 1)], approx_strategy

    def prepare_aggregation_array(self, approx_steps):
        aggregation_array = []
        for problem in range(self.problem_number):
            step_quality_array = []
            for step in range(approx_steps + 1):
                step_quality_array.append([])
            aggregation_array.append(step_quality_array)
        return aggregation_array

    def get_network_solution_for_problem(self, network: nn.Module):
        problem_best_approx_percent_array = []
        for problem in range(self.problem_number):
            one_hot_problem = np.zeros(self.problem_number)
            one_hot_problem[problem] = 1

            tensor_input = torch.from_numpy(one_hot_problem.astype(np.float32))
            tensor_output = network.forward(tensor_input)

            problem_best_approx_percent_array.append(tensor_output.detach().numpy()[0])
        return problem_best_approx_percent_array

    def visualize_results(self, network: nn.Module):
        aggregation_array, percent_array, approx_strategy = self.aggregate_saved_problem_data(self.solver)
        best_approx = self.get_network_solution_for_problem(network)
        solved_approx_quality_array = self.get_approx_quality_pair_array(aggregation_array, percent_array, best_approx)
        approx_quality_graphs_problems_ml(aggregation_array, percent_array, solved_approx_quality_array, approx_strategy, self.solver)


    def get_approx_quality_pair_array(self, aggregation_array: list, percent_array: list,  best_approx: list):
        best_approx_pairs_array = []
        for problem in range(self.problem_number):
            problem_percent_quality = [best_approx[problem], aggregation_array[problem][
                self.get_step_for_approx_percent(best_approx[problem], percent_array)]]
            best_approx_pairs_array.append(problem_percent_quality)
        return best_approx_pairs_array

    def get_step_for_approx_percent(self, perfect_percent: float, percent_array: list):
        return np.argmin([abs(percent_array_value - perfect_percent) for percent_array_value in percent_array])


class Data(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train.astype(np.float32))
        self.y = torch.from_numpy(y_train.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
