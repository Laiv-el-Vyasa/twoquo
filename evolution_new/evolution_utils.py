import time

import numpy as np
import torch

from torch.utils.data import Dataset
from config import load_cfg
from pipeline_util import QUBOGenerator
from recommendation import RecommendationEngine
from typing import Callable

cuda = torch.device('cuda')

cfg = load_cfg(cfg_id='test_evol_m3sat')
qubo_size = cfg['pipeline']['problems']['qubo_size']
problem = cfg['pipeline']['problems']['problems'][0]

solver = 'qbsolv_simulated_annealing'


def get_training_dataset(config: dict) -> dict:
    qubo_list, problem_list = get_problem_qubos(config)
    solutions_list, energy_list = get_qubo_solutions(qubo_list, config)
    return {
        'qubo_list': qubo_list,
        'energy_list': energy_list,
        'solutions_list': solutions_list,
        'problem_list': problem_list
    }


# Solves a list of qubos returning two lists, the first containing the solutions, the second the minimal energies
def get_qubo_solutions(qubo_list: list, config: dict) -> (list[list], list[float]):
    solutions_list = []
    energy_list = []
    for qubo in qubo_list:
        best_solution, solutions, min_energy = solve_qubo(qubo, config)
        solutions_list.append(solutions)
        energy_list.append(min_energy)
    return solutions_list, energy_list


def get_problem_qubos(config: dict):
    generator = QUBOGenerator(config)
    qubos, labels, problems = generator.generate()
    return qubos, problems


# Get quality of the approximated qubo, regarding the quality of the solutions and how much approximation occurred
# Returning quality, best solution and degree of approximation
def get_quality_of_approxed_qubo(qubo: np.array, approxed_qubo: np.array, solutions: np.array,
                                 config: dict, print_solutions=False) -> tuple[float, list, float]:
    absolute_approx_count, relative_approx_count = get_approximation_count(qubo, approxed_qubo)
    _, best_solutions_approx, min_energy_approx = solve_qubo(approxed_qubo, config)
    if print_solutions:
        print(best_solutions_approx)
    min_solution_quality, best_solution_approx = get_min_solution_quality(best_solutions_approx, qubo, solutions)
    return min_solution_quality, best_solution_approx, relative_approx_count


def get_approximation_count(qubo: np.array, approxed_qubo: np.array) -> tuple[int, float]:
    approxed_entries = get_nonzero_count(np.subtract(qubo, approxed_qubo))
    return approxed_entries, approxed_entries / get_nonzero_count(qubo)


def linearize_qubo(qubo):
    linearized_qubo = []
    for i in range(len(qubo)):
        for j in range(i + 1):
            linearized_qubo.append(qubo[i][j])
    return linearized_qubo


def apply_approximation_to_qubo(linearized_approx, qubo):
    approxed_qubo = np.zeros((len(qubo), len(qubo)))
    linear_index = 0
    number_of_true_approx = 0
    for i in range(len(qubo)):
        for j in range(i + 1):
            if linearized_approx[linear_index] > 0:
                approxed_qubo[i][j] = qubo[i][j]
                if not i == j:
                    approxed_qubo[j][i] = qubo[j][i]
            else:
                if not qubo[i][j] == 0:
                    number_of_true_approx += 1
            linear_index += 1
    return approxed_qubo, number_of_true_approx


def get_qubo_approx_mask(linearized_approx, qubo):
    qubo_mask = np.zeros((len(qubo), len(qubo)))
    linear_index = 0
    for i in range(len(qubo)):
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
    for i in range(len(qubo)):
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
    for i in range(len(qubo)):
        for j in range(i):
            if not qubo[i][j] == 0:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
    return adj_matrix


def get_diagonal_of_qubo(qubo):
    diagonal = []
    for i in range(len(qubo)):
        diagonal.append([])
        diagonal[i].append(qubo[i][i])
    return diagonal


def get_mean_of_qubo_line(qubo):
    mean_list = []
    qubo_line_mean_list = []
    for i in range(len(qubo)):
        qubo_line_mean_list.append([])
        line_mean = np.mean(qubo[i])
        qubo_line_mean_list[i].append(line_mean)
        mean_list.append(line_mean)
    mean_of_lines = np.mean(mean_list)
    for i in range(len(qubo)):
        qubo_line_mean_list[i][0] -= mean_of_lines
    return qubo_line_mean_list


def get_tensor_of_structure(ndarray, np_type=np.float32):
    return torch.from_numpy(np.array(ndarray).astype(np_type))


# Returns the best solution / min energy (provided values and order is not always to be trusted)
def solve_qubo(qubo: np.array, config: dict):
    engine = RecommendationEngine(cfg=config)
    metadata = engine.recommend(qubo)
    # print('Metadata solutions: ', metadata.solutions[solver][0])
    # print('Metadata energies: ', metadata.energies[solver][0])
    solution_list = metadata.solutions[solver][0]
    energy_list = metadata.energies[solver][0]
    calc_energies = []
    for solution in solution_list:
        calc_energies.append(solution.dot(qubo.dot(solution)))
    # print('Calc energies: ', calc_energies)
    # print(metadata.solutions[solver], metadata.energies[solver])
    return solution_list[np.argmin(calc_energies)], solution_list, np.min(calc_energies)


# Calculated solution quality of approximation on qubo:
# First calculate the min energy of the original solutions, then of the approxed solutions on the original qubo
# Then calculate solution quality using the results, returning quality and the best approx solution
def get_min_solution_quality(approx_solutions: np.array, qubo: np.array, solutions: np.array) -> tuple[float, list]:
    min_energy_list = []
    for solution in solutions:
        min_energy_list.append(solution.dot(qubo.dot(solution)))
    min_energy = np.min(min_energy_list)
    solution_quality_array = []
    for approx_solution in approx_solutions:
        approx_energy = approx_solution.dot(qubo.dot(approx_solution))
        # if approx_energy < np.min(min_energy_list):
        #    print('APPROX BETTER THAN ORIGINAL')
        #    print('Min energy list: ', min_energy_list)
        #    print('Approx energy: ', approx_energy)
        solution_quality = get_solution_quality(approx_energy, min_energy)
        # print('Solution quality: ', solution_quality)
        solution_quality_array.append(solution_quality)
    return np.min(solution_quality_array), approx_solutions[np.argmin(solution_quality_array)]


def get_solution_quality(energy, min_energy):  # 0: perfect result, 1: worst result
    solution_quality = 1  # Fallback for the worst result
    if min_energy < 0:  # Minimal energy should be negative, else the qubo does not make sense
        if energy > 0:  # Allowing positive energy will lead to wrong results, set to 0
            energy = 0
        solution_quality = (min_energy - energy) / min_energy
    return solution_quality


def get_nonzero_count(nparray: np.array) -> int:
    return np.count_nonzero(nparray)


def construct_fitness_function(function_name: str, fitness_params: dict) -> Callable[[list, list, list], float]:
    return construct_standard_fitness_function(fitness_params)


def construct_standard_fitness_function(fitness_params: dict) -> Callable[[list, list, list], float]:
    def get_new_fitness_value(qubo_list: list, approxed_qubo_list: list, solution_list: list, config: dict) -> float:
        a, b, c, d, min_approx = extract_fitness_params_from_dict(fitness_params)
        fitness_list = []
        for qubo, approximation, solutions in zip(qubo_list, approxed_qubo_list, solution_list):
            solution_quality, best_approx_solution, true_approx_percent = get_quality_of_approxed_qubo(
                qubo, approximation, solutions, config)
            fitness = (a * (1 - solution_quality) +
                       b * (1 - np.square(d - true_approx_percent)) +
                       c * np.floor(1 - solution_quality))
            if not true_approx_percent > min_approx:  # or true_approx_percent == 1:
                fitness = 0
            fitness_list.append(fitness)
        return np.mean(fitness_list)
    return get_new_fitness_value


def extract_fitness_params_from_dict(fitness_params: dict) -> tuple[float, float, float, float, float]:
    a = fitness_params['a']
    b = fitness_params['b']
    c = fitness_params['c']
    d = fitness_params['d']
    z = fitness_params['z']
    return a, b, c, d, z


def get_fitness_value(linearized_approx_list, qubo_list, min_energy_list, solutions_list, fitness_parameters, problems,
                      min_approx=0):
    a, b, c, d = fitness_parameters
    fitness_list = []
    for linearized_approx, qubo, min_energy, solutions, problem in zip(linearized_approx_list, qubo_list,
                                                                       min_energy_list, solutions_list, problems):
        # print('Problem solving: ', problem)
        # print('Sum numbers: ', np.sum(problem['numbers']))
        # print('Max qubo entry: ', np.max(qubo))
        # print('Linearized approx: ', linearized_approx)
        problem_time = time.time()
        solution_quality, best_approx_solution, true_approx_percent = get_quality_of_approxed_qubo(
            linearized_approx, qubo, solutions, cfg)
        # approx_quality = get_approx_number(linearized_approx) / len(linearized_approx)
        # approx_quality = true_approx / len(linearized_approx)
        fitness = (a * (1 - solution_quality) +
                   b * (1 - np.square(d - true_approx_percent)) +
                   c * np.floor(1 - solution_quality))
        # print('Qubo:', qubo)
        # print('Non-Zero:', get_nonzero_count(linearize_qubo(qubo)))
        # print('Problem solving time: ', time.time() - problem_time)
        # print('True approx %: ', true_approx_percent)
        if not true_approx_percent > min_approx:  # or true_approx_percent == 1:
            fitness = 0
        fitness_list.append(fitness)
    return np.mean(fitness_list)


def aggregate_saved_problems(database, true_approx=False):
    # database = engine.get_database()
    qubo_entries = (qubo_size) * (qubo_size + 1) / 2
    aggregation_array = []
    approx_percent_array = []
    for i, (_, metadata) in enumerate(database.iter_metadata()):
        if i == 0:
            aggregation_array = prepare_aggregation_array(qubo_entries)
        print(f'Step {i} in processing database data')
        for idx, step in enumerate(metadata.approx_solution_quality):
            if true_approx:
                # print(metadata.approx_solution_quality)
                idx = get_nearest_bucket(idx, len(metadata.approx_solution_quality), qubo_entries)
                # print('Nearest Bucket ', idx, ' of ', qubo_entries)
            aggregation_array[0][idx].append(metadata.approx_solution_quality[step][solver][0])
            aggregation_array[1][idx].append(metadata.approx_solution_quality[step][solver][1])
            aggregation_array[2][idx].append(metadata.approx_solution_quality[step][solver][2])
    # print(aggregation_array[1])
    for metric, metric_array in enumerate(aggregation_array):
        new_metric_array = []
        for idx, approx_array in enumerate(metric_array):
            # print('Bucket ', idx, ' filling ', len(approx_array))
            if approx_array:
                # aggregation_array[metric][idx] = np.mean(approx_array)
                new_metric_array.append(np.mean(approx_array))
                if metric == 0:
                    approx_percent_array.append(idx / len(metric_array))
        aggregation_array[metric] = new_metric_array
    if true_approx:
        aggregation_array, approx_percent_array = flatten_aggragation_array(aggregation_array, approx_percent_array)
    print(approx_percent_array)
    return aggregation_array, approx_percent_array


def get_nearest_bucket(approx_idx, approx_length, qubo_entries):
    # return int(np.floor((approx_idx / approx_length) * qubo_entries))
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


def prepare_aggregation_array(approx_steps):
    aggregation_array = [[], [], []]
    for i in range(int(approx_steps) + 1):
        aggregation_array[0].append([])
        aggregation_array[1].append([])
        aggregation_array[2].append([])
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


def fitness_params_to_string(fitness_params: dict) -> str:
    fitness_param_string = ''
    for key in fitness_params:
        fitness_param_string = fitness_param_string + f'_{fitness_param_to_string(fitness_params[key])}'
    return fitness_param_string


def fitness_param_to_string(param: float) -> str:
    param_string = str(param)
    param_string = param_string.replace('.', '')
    return param_string


def check_pipeline_necessity(n_problems, database):
    db_count = 0
    for _, metadata in database.iter_metadata():
        db_count += 1
    print('DB count', db_count)
    return n_problems > db_count


def check_solution(original_solutions, approx_solutions, problem):
    solution_value = 1
    if 'numbers' in problem:
        solution_value = check_np_problem(original_solutions, approx_solutions, problem)
    elif 'graph' in problem and 'n_colors' not in problem and 'tsp' not in problem:
        solution_value = check_mc_problem(original_solutions, approx_solutions, problem)
    elif 'graph' in problem and 'n_colors' in problem:
        solution_value = check_gc_problem(approx_solutions, problem)
    elif 'clauses' in problem and 'n_vars' in problem:
        solution_value = check_m3sat_problem(approx_solutions, problem)
    return solution_value


def check_np_problem(original_solutions, approx_solutions, problem):
    solution_value = 1
    original_solution_values = []
    for org_sol in original_solutions:
        original_solution_values.append(check_np_sum(org_sol, problem['numbers']))
    approx_solution_values = []
    for app_sol in approx_solutions:
        approx_solution_values.append(check_np_sum(app_sol, problem['numbers']))
    if np.min(np.abs(approx_solution_values)) > np.min(np.abs(original_solution_values)):
        solution_value = 0
    return solution_value


def check_np_sum(solution, np_numbers):
    np_sum = 0
    for idx, bit_solution in enumerate(solution):
        if bit_solution:
            np_sum += np_numbers[idx]
        else:
            np_sum -= np_numbers[idx]
    return np_sum


def check_mc_problem(original_solutions, approx_solutions, problem):
    solution_value = 1
    graph = problem['graph']
    original_solution_values = []
    for org_sol in original_solutions:
        original_solution_values.append(get_cut_score(org_sol, graph))
    approx_solution_values = []
    for app_sol in approx_solutions:
        approx_solution_values.append(get_cut_score(app_sol, graph))
    if np.max(original_solution_values) > np.max(approx_solution_values):
        solution_value = 0
    return solution_value


def get_cut_score(solution, graph):
    score = 0
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    set_1 = []
    set_2 = []
    for index, i in enumerate(solution):
        if i:
            set_1.append(nodes[index])
        else:
            set_2.append(nodes[index])
    # print(solution)
    # print(set_1, set_2)
    for node_1, node_2 in edges:
        if (node_1 in set_1 and node_2 in set_2) or (node_1 in set_2 and node_2 in set_1):
            score += 1
    # print('Score: ' + str(score))
    return score


def check_gc_problem(approx_solutions, problem):
    solution_value = 0
    graph = problem['graph']
    n_colors = problem['n_colors']
    node_count = len(list(graph.nodes))
    for app_sol in approx_solutions:
        node_colors = get_colors_of_nodes(app_sol, n_colors, node_count)
        if node_colors and check_colors_with_graph(graph, node_colors):
            solution_value = 1
            break
    return solution_value


def get_colors_of_nodes(solution, n_colors, node_count):
    node_colors = [0 for x in range(node_count)]
    for i in range(node_count):
        color_sum = 0
        for j in range(n_colors):
            color_bit = solution[i * n_colors + j]
            if color_bit:
                node_colors[i] = j
                color_sum += 1
        if not color_sum == 1:  # No, or more than one color selected
            node_colors = []
            break
    return node_colors


def check_colors_with_graph(graph, node_colors):
    all_colors_good = True
    for node_1, node_2 in list(graph.edges):
        if node_colors[node_1] == node_colors[node_2]:
            all_colors_good = False
    return all_colors_good


def check_m3sat_problem(approx_solutions, problem):
    clauses = problem['clauses']
    solution_value = 0
    for app_sol in approx_solutions:
        all_clauses_good = True
        for clause in clauses:
            if not evaluate_clause(app_sol, clause):
                all_clauses_good = False
                break
        if all_clauses_good:
            solution_value = 1
            break
    return solution_value


def evaluate_clause(solution, clause):
    clause_bool = False
    for variable, bool in clause:
        if (solution[variable] + int(bool)) % 2 == 0:  # literal in clause is positive
            clause_bool = True
            break
    return clause_bool


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
