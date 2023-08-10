import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.stats as st
from visualisation import approx_quality_graphs_problems_ml
from pipeline import get_max_solution_quality

from recommendation import RecommendationEngine


# *---------------------------------------------------------------------------------- * #
#                                                                                       #
#                       FILES DEPRECATED - CHECK EVOLUTION                              #
#                                                                                       #
# *---------------------------------------------------------------------------------- * #

def get_optimal_approx(approx_dict, approx_steps, solver, lower_bound, percent_bool: bool):
    optimization_array = np.ones(approx_steps)
    for step in range(approx_steps):
        #if step == 0:
        #    quality = approx_dict[step][solver]
        #else:
        quality = approx_dict[str(step)][solver]
        percent = step / approx_steps
        if quality > lower_bound:
            optimization_array[step] = (1 - quality) + (1 - percent)
    optimal_approx = np.max(np.argmin(optimization_array))
    if percent_bool:
        optimal_approx = optimal_approx / approx_steps
    return optimal_approx


def get_step_for_approx_percent(perfect_percent: float, percent_array: list):
    print(perfect_percent)
    nearest_step = np.argmin([abs(percent_array_value - perfect_percent) for percent_array_value in percent_array])
    if perfect_percent < percent_array[nearest_step]:
        nearest_step -= 1
    return nearest_step


def get_approx_percent(approx_qubo):
    result = np.where(approx_qubo == 0)
    return result[0].size / (len(approx_qubo) * len(approx_qubo[0]))


class DatabaseSetup:
    def __init__(self, config, problem_number, solver, min_solution_quality):
        self.eng = RecommendationEngine(cfg=config)
        self.db = self.eng.get_database()
        #self.db = RecommendationEngine().get_database()
        self.problem_number = problem_number
        self.solver = solver
        self.min_solution_quality = min_solution_quality

    def get_data_for_simple_learning(self):
        X_classes = []
        Y_target = []
        for _, metadata in self.db.iter_metadata():
            # print(metadata)
            problem_one_hot = np.zeros(self.problem_number)
            problem_one_hot[metadata.problem] = 1
            X_classes.append(problem_one_hot)
            target_percent = np.zeros(1)
            target_percent[0] = get_optimal_approx(metadata.approx_solution_quality, metadata.approx,
                                                   self.solver, self.min_solution_quality, True)
            Y_target.append(target_percent)
            # Y_target.append(
            #    get_optimal_approx_percent(metadata.approx_solution_quality, metadata.approx, solver, lower_bound))
        # print(np.array(Y_target))
        return np.array(X_classes), np.array(Y_target)

    def get_data_for_learning(self):
        X_classes = []
        Y_target = []
        approx_steps = 0
        for _, metadata in self.db.iter_metadata():
            approx_steps = metadata.approx
            problem_one_hot = np.zeros(self.problem_number)
            problem_one_hot[metadata.problem] = 1
            X_classes.append(problem_one_hot)
            solution_qualities = []
            for step in range(approx_steps):
                solution_quality = 0#np.zeros(1)
                #if step == 0:
                #    solution_quality = metadata.approx_solution_quality[step][self.solver]
                #else:
                solution_quality = metadata.approx_solution_quality[str(step)][self.solver]

                solution_qualities.append(solution_quality)
            Y_target.append(np.array(solution_qualities))
            # Y_target.append(
            #    get_optimal_approx_percent(metadata.approx_solution_quality, metadata.approx, solver, lower_bound))
        # print(np.array(Y_target))
        return np.array(X_classes), np.array(Y_target), approx_steps

    def get_data_for_simple_classification(self):
        X_classes = []
        Y_classes = []
        approx_steps = 0
        step_problem_array = []
        for i in range(self.problem_number):
            step_problem_array.append({})
        for _, metadata in self.db.iter_metadata():
            approx_steps = metadata.approx
            problem_one_hot = np.zeros(self.problem_number)
            problem_one_hot[metadata.problem] = 1
            X_classes.append(problem_one_hot)
            optimal_approx = get_optimal_approx(metadata.approx_solution_quality, metadata.approx,
                                                self.solver, self.min_solution_quality, False)
            if not optimal_approx in step_problem_array[metadata.problem].keys():
                step_problem_array[metadata.problem][optimal_approx] = 1
            else:
                step_problem_array[metadata.problem][optimal_approx] += 1
            Y_classes.append(optimal_approx)
        for i in range(self.problem_number):
            print(f'Problem {i}:', dict(sorted(step_problem_array[i].items())))

        return np.array(X_classes), np.array(Y_classes), approx_steps

    def get_data_for_enc_dec(self):
        X_qubos = []
        Y_energies = []
        for _, metadata in self.db.iter_metadata():
            X_qubos.append(metadata.Q)
            Y_energies.append((metadata.best_energies[self.solver], metadata.worst_energy))
        return X_qubos, Y_energies

    def get_solution_quality_of_approxed(self, original_qubo, approx_qubo, best_energy, worst_energy):
        approx_metadata = self.eng.recommend(approx_qubo)
        return get_max_solution_quality(approx_metadata.solutions, original_qubo, best_energy, worst_energy)

    def aggregate_saved_problem_data(self, solver):
        global approx_steps
        approx_strategy = False
        aggregation_array = []
        problem_collection = set()
        for i, (_, metadata) in enumerate(self.db.iter_metadata()):
            approx_steps = metadata.approx
            problem_collection.add(metadata.problem)
            if i == 0:
                aggregation_array = self.prepare_aggregation_array(approx_steps)
                approx_strategy = metadata.approx_strategy
            for step in range(approx_steps):
                #if step == 0:
                #    quality = metadata.approx_solution_quality[step][solver]
                #else:
                quality = metadata.approx_solution_quality[str(step)][solver]
                aggregation_array[metadata.problem][step].append(quality)
        print(problem_collection)
        print(len(aggregation_array))
        for problem, problem_array in enumerate(aggregation_array):
            for approx, approx_array in enumerate(problem_array):
                aggregation_array[problem][approx] = np.mean(approx_array)
        return aggregation_array, [n / approx_steps for n in
                                                    range(approx_steps)], approx_strategy

    def prepare_aggregation_array(self, approx_steps):
        aggregation_array = []
        for problem in range(self.problem_number):
            step_quality_array = []
            for step in range(approx_steps):
                step_quality_array.append([])
            aggregation_array.append(step_quality_array)
        return aggregation_array

    def get_network_solution_for_problem(self, network: nn.Module, precent_array, learner_type: str):
        problem_best_approx_step_array = []
        softmax = nn.Softmax(dim=0)
        for problem in range(self.problem_number):
            one_hot_problem = np.zeros(self.problem_number)
            one_hot_problem[problem] = 1

            tensor_input = torch.from_numpy(one_hot_problem.astype(np.float32))
            tensor_output = network.forward(tensor_input)

            if learner_type == 'classification':
                print(tensor_output)
                tensor_output = softmax(tensor_output)

            detached_output = tensor_output.detach().numpy()

            if learner_type == 'classification':
                problem_best_approx_step_array.append(np.argmax(detached_output))
            elif learner_type == 'simple_learner':
                problem_best_approx_step_array.append(get_step_for_approx_percent(detached_output[0], precent_array))
            elif learner_type == 'learner':
                problem_best_approx_step_array.append(self.get_learned_optimal_step(detached_output))
        return problem_best_approx_step_array

    def get_learned_optimal_step(self, learned_array):
        optimal_step = 0
        eligible_steps = [i for i in range(len(learned_array)) if learned_array[i] > self.min_solution_quality]
        if eligible_steps:
            optimal_step = np.max(eligible_steps)
        return optimal_step

    def visualize_results(self, network: nn.Module, learner_type: str):
        aggregation_array, percent_array, approx_strategy = self.aggregate_saved_problem_data(self.solver)
        best_approx_steps = self.get_network_solution_for_problem(network, percent_array, learner_type)
        solved_approx_quality_array = self.get_approx_quality_pair_array(aggregation_array, percent_array,
                                                                         best_approx_steps)
        approx_quality_graphs_problems_ml(aggregation_array, percent_array, solved_approx_quality_array,
                                          approx_strategy,
                                          self.solver, self.min_solution_quality, learner_type)

    def get_approx_quality_pair_array(self, aggregation_array: list, percent_array: list, best_approx_steps: list):
        best_approx_pairs_array = []
        for problem in range(self.problem_number):
            problem_percent_quality = [percent_array[best_approx_steps[problem]], aggregation_array[problem][
                best_approx_steps[problem]]]
            best_approx_pairs_array.append(problem_percent_quality)
        return best_approx_pairs_array


class Data(Dataset):
    def __init__(self, X_train, y_train, enc_dec=False):
        if not enc_dec:
            self.X = torch.from_numpy(X_train.astype(np.float32))
            self.y = torch.from_numpy(y_train.astype(np.float32))
        else:
            self.X = torch.tensor(X_train)
            self.y = y_train
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
