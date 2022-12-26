import time

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from config import load_cfg
from pipeline_util import QUBOGenerator
from recommendation import RecommendationEngine

cuda = torch.device('cuda')

cfg = load_cfg(cfg_id='test_evol_large')
qubo_size = cfg['pipeline']['problems']['qubo_size']
problem = cfg['pipeline']['problems']['problems'][0]
engine = RecommendationEngine(cfg=cfg)
generator = QUBOGenerator(cfg)

solver = 'qbsolv_simulated_annealing'


def get_training_dataset(n_problems, include_loops=True):
    linearized_qubo_list, qubo_list, energy_list, \
    solution_list, problem_list, edge_index_list, \
        edge_weight_list = get_linearized_qubos(n_problems, include_loops=include_loops)
    return torch.from_numpy(np.array(linearized_qubo_list).astype(np.float32)), qubo_list, \
           energy_list, solution_list, problem_list, edge_index_list, edge_weight_list


def get_linearized_qubos(n_problems, include_loops=True):
    qubo_list, problem_list = get_problem_qubos(n_problems)
    linarized_qubo_list = []
    energy_list = []
    solution_List = []
    edge_index_list = []
    edge_weight_list = []
    for qubo in qubo_list:
        solutions, min_energy = solve_qubo(qubo)
        linarized_qubo_list.append(linearize_qubo(qubo))
        edge_index, edge_weights = get_edge_data(qubo, include_loops=include_loops)
        edge_index_list.append(edge_index)
        edge_weight_list.append(edge_weights)
        energy_list.append(min_energy)
        solution_List.append(solutions[0])
    return linarized_qubo_list, qubo_list, energy_list, solution_List, problem_list, edge_index_list, edge_weight_list


def get_quality_of_approxed_qubo(linearized_approx, qubo, min_energy, print_solutions=False):
    approxed_qubo, true_approx = apply_approximation_to_qubo(linearized_approx, qubo)
    solutions, min_energy_approx = solve_qubo(approxed_qubo)
    if print_solutions:
        print(solutions)
    return get_min_solution_quality(solutions, qubo, min_energy), true_approx, \
           true_approx / get_nonzero_count(linearize_qubo(qubo))


def linearize_qubo(qubo):
    linearized_qubo = []
    for i in range(qubo_size):
        for j in range(i + 1):
            linearized_qubo.append(qubo[i][j])
    return linearized_qubo


def apply_approximation_to_qubo(linearized_approx, qubo):
    approxed_qubo = np.zeros((qubo_size, qubo_size))
    linear_index = 0
    number_of_true_approx = 0
    for i in range(qubo_size):
        for j in range(i + 1):
            if linearized_approx[linear_index] > 0:
                approxed_qubo[i][j] = qubo[i][j]
                if not i == j:
                    approxed_qubo[j][i] = qubo[j][i]
            else:
                if not qubo[i][j] == 0:
                    number_of_true_approx += 1
            linear_index += 1
    #print('Approxed qubo: ', approxed_qubo.tolist())
    return approxed_qubo, number_of_true_approx


def get_qubo_approx_mask(linearized_approx):
    qubo_mask = np.zeros((qubo_size, qubo_size))
    linear_index = 0
    for i in range(qubo_size):
        for j in range(i + 1):
            if linearized_approx[linear_index] > 0:
                qubo_mask[i][j] = 1
                if not i == j:
                    qubo_mask[j][i] = 1
            linear_index += 1
    return qubo_mask


def get_edge_data(qubo, include_loops=True):
    edge_index = [[], []]
    edge_weight = []
    for i in range(qubo_size):
        for j in range(i + 1):
            if not i == j or include_loops:
                if not qubo[i][j] == 0:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_weight.append(qubo[i][j])
                    if not i == j:
                        edge_index[0].append(j)
                        edge_index[1].append(i)
                        edge_weight.append(qubo[i][j])
    return edge_index, edge_weight


def get_adjacency_matrix_of_qubo(qubo):
    adj_matrix = np.zeros(np.shape(qubo))
    for i in range(qubo_size):
        for j in range(i):
            if not qubo[i][j] == 0:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
    return adj_matrix


def get_diagonal_of_qubo(qubo):
    diagonal = []
    for i in range(qubo_size):
        diagonal.append([])
        diagonal[i].append(qubo[i][i])
    return diagonal


def get_tensor_of_structure(ndarray, np_type=np.float32):
    return torch.from_numpy(np.array(ndarray).astype(np_type))


def get_problem_qubos(n_problems):
    qubos, labels, problems = generator.generate()
    return qubos[:n_problems], problems[:n_problems]


def solve_qubo(qubo):
    metadata = engine.recommend(qubo)
    # print(metadata.solutions[solver], metadata.energies[solver])
    return metadata.solutions[solver][0], np.min(metadata.energies[solver][0])


def get_min_solution_quality(solutions, qubo, min_energy):
    solution_quality_array = []
    for solution in solutions:
        solution_quality = get_solution_quality(solution.dot(qubo.dot(solution)), min_energy)
        solution_quality_array.append(solution_quality)
    return np.min(solution_quality_array), solutions[np.argmin(solution_quality_array)]


def get_solution_quality(energy, min_energy):
    return (min_energy - energy) / min_energy


def get_nonzero_count(nparray):
    # print('Linearized approx, to calc approx_qual: ', nparray)
    return np.count_nonzero(nparray)


def get_fitness_value(linearized_approx_list, qubo_list, min_energy_list, fitness_parameters, problems, min_approx=0):
    a, b, c, d = fitness_parameters
    fitness_list = []
    for linearized_approx, qubo, min_energy, problem in zip(linearized_approx_list, qubo_list, min_energy_list, problems):
        #print('Problem solving: ', problem)
        #print('Sum numbers: ', np.sum(problem['numbers']))
        #print('Max qubo entry: ', np.max(qubo))
        #print('Linearized approx: ', linearized_approx)
        problem_time = time.time()
        (solution_quality, best_approx_solution), true_approx, true_approx_percent = get_quality_of_approxed_qubo(linearized_approx, qubo,
                                                                                          min_energy)
        # approx_quality = get_approx_number(linearized_approx) / len(linearized_approx)
        # approx_quality = true_approx / len(linearized_approx)
        # fitness_list.append((1 - solution_quality) * approx_quality)
        fitness = (a * (1 - solution_quality) +
                   b * (1 - np.square(d - true_approx_percent)) +
                   c * np.floor(1 - solution_quality))
        # print('Qubo:', qubo)
        # print('Non-Zero:', get_nonzero_count(linearize_qubo(qubo)))
        #print('Problem solving time: ', time.time() - problem_time)
        if not true_approx_percent > min_approx:
            fitness = 0
        fitness_list.append(fitness)
    return np.mean(fitness_list)


def aggregate_saved_problems(true_approx=False):
    database = engine.get_database()
    qubo_entries = (qubo_size) * (qubo_size + 1) / 2
    aggregation_array = []
    approx_percent_array = []
    for i, (_, metadata) in enumerate(database.iter_metadata()):
        if i == 0:
            aggregation_array = prepare_aggregation_array(qubo_entries)
        for idx, step in enumerate(metadata.approx_solution_quality):
            if true_approx:
                #print(metadata.approx_solution_quality)
                idx = get_nearest_bucket(idx, len(metadata.approx_solution_quality), qubo_entries)
                #print('Nearest Bucket ', idx, ' of ', qubo_entries)
            aggregation_array[0][idx].append(metadata.approx_solution_quality[step][solver][0])
            aggregation_array[1][idx].append(metadata.approx_solution_quality[step][solver][1])
    #print(aggregation_array[1])
    for metric, metric_array in enumerate(aggregation_array):
        new_metric_array = []
        for idx, approx_array in enumerate(metric_array):
            #print('Bucket ', idx, ' filling ', len(approx_array))
            if approx_array:
                # aggregation_array[metric][idx] = np.mean(approx_array)
                new_metric_array.append(np.mean(approx_array))
                if metric == 0:
                    approx_percent_array.append(idx / len(metric_array))
        aggregation_array[metric] = new_metric_array
    if true_approx:
        aggregation_array, approx_percent_array = flatten_aggragation_array(aggregation_array, approx_percent_array)
    return aggregation_array, approx_percent_array


def get_nearest_bucket(approx_idx, approx_length, qubo_entries):
    #return int(np.floor((approx_idx / approx_length) * qubo_entries))
    return int(np.ceil((approx_idx / approx_length) * qubo_entries))


def flatten_aggragation_array(aggregation_array, approx_percent_array):
    new_aggragation_array = []
    for metric in aggregation_array:
        new_aggragation_array.append([])
    new_approx_percent_array = []
    for idx in range(len(approx_percent_array) - 1):
        if idx == 0:
            new_approx_percent_array.append(approx_percent_array[idx])
            for metric, metric_array in enumerate(aggregation_array):
                new_aggragation_array[metric].append(metric_array[idx])
        else:
            new_approx_percent_array.append(
                np.mean([approx_percent_array[idx - 1], approx_percent_array[idx], approx_percent_array[idx + 1]])
            )
            for metric, metric_array in enumerate(aggregation_array):
                new_aggragation_array[metric].append(
                    np.mean([metric_array[idx - 1], metric_array[idx], metric_array[idx + 1]])
                )
    return new_aggragation_array, new_approx_percent_array


def prepare_aggregation_array(qubo_entries):
    aggregation_array = [[], []]
    for i in range(int(qubo_entries) + 1):
        aggregation_array[0].append([])
        aggregation_array[1].append([])
    return aggregation_array


def check_model_config_fit(model_descr, independence):
    parameter_list = model_descr.split("_")
    model_problem = parameter_list[2]
    model_size = int(parameter_list[3])
    # fitness_params = (get_param_value(parameter_list[3]),
    #                  get_param_value(parameter_list[4]),
    #                  get_param_value(parameter_list[5]),
    #                  get_param_value(parameter_list[6]))
    return (model_problem == problem or independence) and (model_size == qubo_size or independence)  # , fitness_params


def get_param_value(param_string):
    return_value = 0
    if param_string.startswith("0"):
        return_value = float(f"0.{param_string[1:]}")
    else:
        return_value = int(param_string)
    return return_value


def check_pipeline_necessity(n_problems):
    database = engine.get_database()
    db_count = 0
    for _, metadata in database.iter_metadata():
        db_count += 1
    print('DB count', db_count)
    return n_problems > db_count


def check_solution(solution, problem):
    solution_value = 0
    if 'numbers' in problem:
        solution_value = check_np_sum(solution, problem['numbers'])
    return solution_value


def check_np_sum(solution, np_numbers):
    np_sum = 0
    for idx, bit_solution in enumerate(solution):
        if bit_solution:
            np_sum += np_numbers[idx]
        else:
            np_sum -= np_numbers[idx]
    return np_sum


class Data(Dataset):
    def __init__(self, x_train, x_qubos, x_energy):
        self.X = torch.from_numpy(x_train.astype(np.float32))
        self.qubos = x_qubos
        self.energy = x_energy
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.qubos[index], self.energy[index]

    def __len__(self):
        return self.len
