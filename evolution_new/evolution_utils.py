import random
import time

import networkx
import numpy as np
from numpy import random as np_random
import torch
import os
import shutil

from torch.utils.data import Dataset
from config import load_cfg
from evolution_new.new_visualisation import plot_cities_with_solution, bruteforce_verification, qubo_heatmap, \
    get_best_solution
from pipeline_util import QUBOGenerator
from recommendation import RecommendationEngine
from typing import Callable

cuda = torch.device('cuda')

cfg = load_cfg(cfg_id='test_evol_m3sat')
qubo_size = cfg['pipeline']['problems']['qubo_size']
problem = cfg['pipeline']['problems']['problems'][0]

solver = 'qbsolv_simulated_annealing'

check_tsp = False


def delete_data():
    try:
        os.chdir('../data/')
        current_dirctory = os.getcwd()
        for data in os.listdir(current_dirctory):
            shutil.rmtree(current_dirctory + '/' + data)
        os.chdir('../evolution_new/')
    except OSError as e:
        print("Error: %s" % e.strerror)


def get_file_name(base_name: str, config: dict, fitness_params: dict, analysis=False, steps=100) -> str:
    name = base_name
    problems = config['pipeline']['problems']['problems']
    for prob in problems:
        name = name + '_' + prob
    name = name + '_' + str(config['pipeline']['problems']['qubo_size'])
    if analysis:
        name = name + '_' + str(steps) + '_steps'
    else:
        name = name + fitness_params_to_string(fitness_params)
    return name


def get_analysis_results_file_name(base_name: str, training_config: dict, config: dict, fitness_params: dict) -> str:
    name = base_name
    problems_train = training_config['pipeline']['problems']['problems']
    for prob in problems_train:
        name = name + '_' + prob
    name = name + '_' + str(training_config['pipeline']['problems']['qubo_size'])
    name = name + fitness_params_to_string(fitness_params)
    name = name + '_eval'
    problems_eval = config['pipeline']['problems']['problems']
    for prob2 in problems_eval:
        name = name + '_' + prob2
    name = name + '_' + str(config['pipeline']['problems']['qubo_size'])
    return name


def fitness_params_to_string(fitness_params: dict) -> str:
    fitness_param_string = ''
    for key in fitness_params:
        fitness_param_string = fitness_param_string + f'_{fitness_param_to_string(fitness_params[key])}'
    return fitness_param_string


def fitness_param_to_string(param: float) -> str:
    param_string = str(param)
    param_string = param_string.replace('.', '')
    return param_string


def check_tsp_solutions(qubo_list: list, problem_list: list, solutions_list: list):
    for i in range(len(problem_list)):
        dist1, best_path1 = bruteforce_verification(problem_list[i]['cities'], problem_list[i]['dist_matrix'])
        dist2, best_path2 = get_best_solution(solutions_list[i], problem_list[i]['dist_matrix'])
        if not dist2 == dist1:
            print(dist1, dist2)
            print("WRONG SOLUTION FOUND")
            print(solutions_list[i])
            print(problem_list[i])
            plot_cities_with_solution(problem_list[i]['cities'], best_path1)
            plot_cities_with_solution(problem_list[i]['cities'], best_path2)
            print(qubo_list[i])
            qubo_heatmap(qubo_list[i])


def get_training_dataset(config: dict) -> dict:
    qubo_list, problem_list = get_problem_qubos(config)
    solutions_list, energy_list = get_qubo_solutions(qubo_list, config)
    # print(problem_list)
    # qubo_heatmap(qubo_list[0])
    if 'tsp' in problem_list[0] and check_tsp:
        check_tsp_solutions(qubo_list, problem_list, solutions_list)
    problem_dict = {
        'qubo_list': qubo_list,
        'energy_list': energy_list,
        'solutions_list': solutions_list,
        'problem_list': problem_list
    }
    if "scale" in config['pipeline']['problems']:
        problem_dict["scale_list"] = get_random_scale_list(config)
    # print('Scale list, ', problem_dict["scale_list"])
    return problem_dict


def get_random_scale_list(config: dict) -> list[float]:
    min_sc, max_sc = config['pipeline']['problems']['scale']['min'], config['pipeline']['problems']['scale']['max']
    n_problems = config['pipeline']['problems']['n_problems']
    rng = np.random.default_rng()
    return rng.uniform(min_sc, max_sc, n_problems)


# Solves a list of qubos returning two lists, the first containing the solutions, the second the minimal energies
def get_qubo_solutions(qubo_list: list, config: dict) -> (list[list], list[float]):
    solutions_list = []
    energy_list = []
    i = 0
    for qubo in qubo_list:
        # print('Solving Qubo ' + str(i))
        i += 1
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
                                 config: dict, print_solutions=False) \
        -> tuple[float, list, float, float, float, float, int]:
    absolute_approx_count, relative_approx_count = get_approximation_count(qubo, approxed_qubo)
    _, best_solutions_approx, min_energy_approx = solve_qubo(approxed_qubo, config)
    if print_solutions:
        print(best_solutions_approx)
    min_solution_quality, best_solution_approx, mean_solution_quality, min_mean_sol_qual, mean_mean_sol_qual \
        = get_min_solution_quality(best_solutions_approx, qubo, solutions)
    return min_solution_quality, best_solution_approx, relative_approx_count, mean_solution_quality, \
           min_mean_sol_qual, mean_mean_sol_qual, absolute_approx_count


def get_approximation_count(qubo: np.array, approxed_qubo: np.array) -> tuple[int, float]:
    # approxed_entries = get_nonzero_count(np.subtract(np.triu(qubo), np.triu(approxed_qubo)))
    # return approxed_entries, approxed_entries / get_nonzero_count(np.triu(qubo))
    approxed_entries = get_nonzero_count(np.subtract(np.subtract(np.triu(qubo), np.diag(np.diag(qubo))),
                                                     np.subtract(np.triu(approxed_qubo),
                                                                 np.diag(np.diag(approxed_qubo)))))
    return approxed_entries, approxed_entries / get_nonzero_count(np.subtract(np.triu(qubo), np.diag(np.diag(qubo))))


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


def get_qubo_approx_mask(approxed_qubo: list, qubo: list) -> list:
    qubo_mask = np.zeros((len(qubo), len(qubo)))
    for i in range(len(qubo)):
        for j in range(i + 1):
            if not qubo[i][j] == 0 and approxed_qubo[i][j] == 0:
                qubo_mask[i][j] = 1
                if not i == j:
                    qubo_mask[j][i] = 1
    return qubo_mask


def get_small_qubo(qubo: list, n_colors: int) -> list:
    if len(qubo) % n_colors != 0:
        print('QUBO not reducabe!')
        return qubo
    new_size = int(len(qubo) / n_colors)
    small_qubo = np.zeros((new_size, new_size))
    for i in range(new_size):
        for j in range(new_size):
            small_qubo[i][j] = qubo[i * n_colors][j * n_colors]
    return small_qubo


def get_reducability_number(prob: dict) -> int:
    n = 0
    if 'n_colors' in prob:
        n = prob['n_colors']
    if 'tsp' in prob:
        n = len(prob['cities'])
    return n


def remove_hard_constraits_from_qubo(qubo: list, problem: dict, remove_diagonal: bool) -> list:
    return_qubo = np.zeros((len(qubo), len(qubo)))
    return_qubo += qubo
    if 'tsp' in problem:
        n = len(problem['cities'])
        # Remove sub-diagonals
        for i in range(n):
            for j in range(i):
                for k in range(n):
                    return_qubo[n * i + k][n * j + k] = 0
                    return_qubo[n * j + k][n * i + k] = 0
        # Remove triangles over main diagonal
        for i in range(n):
            for j in range(n):
                for k in range(j):
                    return_qubo[i * n + j][i * n + k] = 0
                    return_qubo[i * n + k][i * n + j] = 0
    elif 'n_colors' in problem:
        n = problem['n_colors']
        m = int(len(qubo) / n)
        # Remove triangles over main diagonal
        for i in range(m):
            for j in range(n):
                for k in range(j):
                    return_qubo[i * n + j][i * n + k] = 0
                    return_qubo[i * n + k][i * n + j] = 0
    # Remove diagonal
    if remove_diagonal:
        return_qubo -= np.diag(np.diagonal(qubo))
    return return_qubo


def get_edge_data(qubo: list, include_loops=True) -> tuple[list[list, list], list]:
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


def get_diagonal_of_qubo(qubo: list) -> list:
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


def get_min_of_tsp_qubo_line_normalized(qubo: list, n: int) -> list:
    min_distance_list = [[] for _ in range(n)]
    distance_collection_list = [[] for _ in range(n)]
    # Collect distances
    for i in range(n):
        for j in range(len(qubo)):
            # Do not look at diagonal quadrant, all values here are set
            # Do not look at diagonal in sub quadrant
            # Skip the zeros
            if not i * n <= j < (i + 1) * n \
                    and not j % n == 0 \
                    and not qubo[i * n][j] == 0:
                distance_collection_list[i].append(qubo[i * n][j])
    # Set distances and get min distance
    max_min_distance = 0
    for i in range(n):
        line_min_dist = np.min(distance_collection_list[i])
        if line_min_dist > max_min_distance:
            max_min_distance = line_min_dist
        min_distance_list[i].append(line_min_dist)
    # Normalize distances: 1 := longest shortest distance
    for i in range(n):
        min_distance_list[i][0] = (min_distance_list[i][0] / max_min_distance) * n
    return min_distance_list


def get_min_of_tsp_qubo_line_normalized_onehot(qubo: list, n: int) -> list:
    min_distance_list = [[] for _ in range(len(qubo))]
    distance_collection_list = [[] for _ in range(len(qubo))]
    # Collect distances
    for i in range(len(qubo)):
        for j in range(len(qubo)):
            # Do not look at diagonal quadrant, all values here are set
            # Do not look at diagonal in sub quadrant
            # Skip the zeros
            n_ = np.floor(i / n)
            k_ = i % n
            if not n_ * n <= j < (n_ * n) + n \
                    and not j % n == k_ \
                    and not qubo[i][j] == 0:
                distance_collection_list[i].append(qubo[i][j])

    # Set distances and get min distance
    # print('Dist collection list: ', distance_collection_list)
    max_min_distance = 0
    for i in range(len(qubo)):
        line_min_dist = np.mean(np.sort(distance_collection_list[i])[:3])
        if line_min_dist > max_min_distance:
            max_min_distance = line_min_dist
        min_distance_list[i].append(line_min_dist)
    # Normalize distances: 1 := longest shortest distance
    for i in range(len(qubo)):
        min_distance_list[i][0] = (min_distance_list[i][0] / max_min_distance) * n
    return min_distance_list


def get_number_of_edges_for_gc(qubo: list, n: int):
    line_edge_list = [[] for _ in range(len(qubo))]
    for i in range(len(qubo)):
        edge_count = 0
        for j in range(len(qubo)):
            n_ = np.floor(i / n)
            if not n_ * n <= j < (n_ * n) + n \
                    and qubo[i][j] != 0:
                edge_count += 1
        line_edge_list[i].append(edge_count)
    # print(line_edge_list)
    return line_edge_list


def get_tensor_of_structure(ndarray, np_type=np.float32):
    return torch.from_numpy(np.array(ndarray).astype(np_type))


def matrix_to_qubo(qubo_matrix: np.array) -> dict:
    qubo = {}
    for pos, val in np.ndenumerate(qubo_matrix):
        if val:
            qubo[pos] = val
    return qubo


def qubo_to_matrix(qubo: dict, size: int) -> np.array:
    Q = np.zeros((size, size))
    for (x, y), v in qubo.items():
        Q[x][y] = v
    return Q


# Returns the best solution / min energy (provided values and order is not always to be trusted)
def solve_qubo(qubo: np.array, config: dict):
    engine = RecommendationEngine(cfg=config)
    metadata = engine.recommend(qubo)
    # print('Metadata solutions: ', metadata.solutions[solver][0])
    # print('Metadata energies: ', metadata.energies[solver][0])
    solution_list = []
    calc_energies = []
    for solutions in metadata.solutions[solver]:
        solution_list.append(solutions[0])  # Append first solution given by the solver
        calc_energies.append(solutions[0].dot(qubo.dot(solutions[0])))
    # print(solution_list)
    # for solution in solution_list:
    #    calc_energies.append(solution.dot(qubo.dot(solution)))
    # print('Calc energies: ', calc_energies)
    # print(metadata.solutions[solver], metadata.energies[solver])
    return solution_list[np.argmin(calc_energies)], solution_list, np.min(calc_energies)


# Calculated solution quality of approximation on qubo:
# First calculate the min energy of the original solutions, then of the approxed solutions on the original qubo
# Then calculate solution quality using the results, returning quality and the best approx solution
def get_min_solution_quality(approx_solutions: np.array, qubo: np.array, solutions: np.array) \
        -> tuple[float, list, float, float, float]:
    min_energy_list = []
    for solution in solutions:
        min_energy_list.append(solution.dot(qubo.dot(solution)))
    min_energy = np.min(min_energy_list)
    mean_energy = np.mean(min_energy_list)
    solution_quality_array = []
    mean_solution_quality_array = []
    for approx_solution in approx_solutions:
        approx_energy = approx_solution.dot(qubo.dot(approx_solution))
        # if approx_energy < np.min(min_energy_list):
        #    print('APPROX BETTER THAN ORIGINAL')
        #    print('Min energy list: ', min_energy_list)
        #    print('Approx energy: ', approx_energy)
        solution_quality = get_solution_quality(approx_energy, min_energy)
        # print('Solution quality: ', solution_quality)
        solution_quality_array.append(solution_quality)
        mean_solution_quality_array.append(get_solution_quality(approx_energy, mean_energy))
    # print('Solution quality array', solution_quality_array)
    return np.min(solution_quality_array), approx_solutions[np.argmin(solution_quality_array)], \
           np.mean(solution_quality_array), np.min(mean_solution_quality_array), np.mean(mean_solution_quality_array)


def get_solution_quality(energy, min_energy):  # 0: perfect result, 1: worst result
    solution_quality = 1  # Fallback for the worst result
    if min_energy < 0:  # Minimal energy should be negative, else the qubo does not make sense
        if energy > 0:  # Allowing positive energy will lead to wrong results, set to 0
            energy = 0
        solution_quality = (min_energy - energy) / min_energy
    return solution_quality


def get_nonzero_count(nparray: np.array) -> int:
    return np.count_nonzero(nparray)


def construct_fitness_function(function_name: str, fitness_params: dict) -> Callable[[dict, dict], float]:
    return construct_standard_fitness_function(fitness_params)


def construct_standard_fitness_function(fitness_params: dict) -> Callable[[dict, dict], float]:
    def get_new_fitness_value(problem_dict: dict, config: dict) -> float:
        a, b, c, d, min_approx = extract_fitness_params_from_dict(fitness_params)
        fitness_list = []
        qubo_list, approxed_qubo_list, solution_list = problem_dict['qubo_list'], problem_dict['approxed_qubo_list'], \
                                                       problem_dict['solutions_list']
        for qubo, approximation, solutions in zip(qubo_list, approxed_qubo_list, solution_list):
            solution_quality, best_approx_solution, true_approx_percent, *_ = get_quality_of_approxed_qubo(
                qubo, approximation, solutions, config)
            fitness = (a * (1 - solution_quality) +
                       b * (1 - np.square(d - true_approx_percent)) +
                       c * np.floor(1 - solution_quality))
            if not true_approx_percent > min_approx:  # or true_approx_percent == 1:
                fitness = 0
            fitness_list.append(fitness)
        return np.mean(fitness_list)

    return get_new_fitness_value


def construct_scale_fitness_function(fitness_params: dict) -> Callable[[dict, dict], float]:
    def get_new_fitness_value(problem_dict: dict, config: dict) -> float:
        a, b, c, _, min_approx = extract_fitness_params_from_dict(fitness_params)
        fitness_list = []
        qubo_list, approxed_qubo_list, \
        solution_list, scale_list = problem_dict['qubo_list'], problem_dict['approxed_qubo_list'], \
                                    problem_dict['solutions_list'], problem_dict['scale_list']
        for qubo, approximation, solutions, scale in zip(qubo_list, approxed_qubo_list, solution_list, scale_list):
            solution_quality, best_approx_solution, true_approx_percent, *_ = get_quality_of_approxed_qubo(
                qubo, approximation, solutions, config)
            # print(scale, true_approx_percent)
            fitness = (a * (1 - solution_quality) +
                       b * (1 - np.square(scale - true_approx_percent)) +
                       c * np.floor(1 - solution_quality))
            if not np.abs(scale - true_approx_percent) < min_approx or true_approx_percent == 0:
                fitness = 0
            # print(fitness)
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


# Return a float between 0 (only the smallest entries have benn approxed) and 1 (the biggest entries have been approxed)
def get_relative_size_of_approxed_entries(approxed_qubo: list, qubo: list) -> float:
    approxed_qubo = np.triu(approxed_qubo)
    qubo = np.triu(qubo)
    approxed_entries_qubo = np.subtract(qubo, approxed_qubo)
    approxed_entry_count = get_nonzero_count(approxed_entries_qubo)
    if approxed_entry_count > 0:
        n_extreme_min = get_n_extreme_entries(approxed_entry_count, qubo, True)
        min_sum = get_sum_of_array(n_extreme_min)
        n_extreme_max = get_n_extreme_entries(approxed_entry_count, qubo, False)
        max_sum = get_sum_of_array(n_extreme_max)
        flat_qubo = np.abs(approxed_entries_qubo.flatten())
        actual_sum = get_sum_of_array(flat_qubo)
        return (actual_sum - min_sum) / (max_sum - min_sum)
    else:
        return 0


def get_n_extreme_entries(n: int, qubo: list, ascending: bool) -> list:
    abs_qubo = np.abs(qubo)
    flat_qubo = abs_qubo.flatten()
    if ascending:
        flat_qubo = np.sort(flat_qubo)
    else:
        flat_qubo[::-1].sort()
    flat_qubo = np.trim_zeros(flat_qubo)
    return flat_qubo[:n]


def get_sum_of_array(array: np.ndarray) -> float:
    return sum(array)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                     Algorithms to generate approximate solutions                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_classical_solution_qualities(solutions: list, qubo: np.array, problem: dict, reads: int,
                                     random_solutions: bool) -> tuple[float, float]:
    classical_solutions = get_classical_solutions(problem, reads, random_solutions)
    min_solution_quality, _, mean_solution_quality, *_ = get_min_solution_quality(classical_solutions, qubo, solutions)
    return min_solution_quality, mean_solution_quality


# Calculating solutions for problems using classical heuristics as well as at random
def get_classical_solutions(problem: dict, reads: int, random_solutions: bool) -> list[list]:
    solution_list = []
    for _ in range(reads):
        if 'cities' in problem:
            if random_solutions:
                solution_list.append(get_random_city_order_solution(problem))
            else:
                solution_list.append(get_classical_solution_tsp(problem))
        elif 'numbers' in problem:
            if random_solutions:
                solution_list.append(get_random_solution_assignment(len(problem['numbers'])))
            else:
                solution_list.append(get_random_differencing_solution(problem))
        elif 'graph' in problem and 'n_colors' not in problem:
            if random_solutions:
                solution_list.append(get_random_solution_assignment(len(problem['graph'].nodes)))
            else:
                solution_list.append(get_solution_conditional_assignments_mc(problem))
        elif 'subset_matrix' in problem:
            if random_solutions:
                solution_list.append(get_random_solution_assignment(len(problem['subset_matrix'][0])))
            else:
                solution_list.append(get_dlx_solution_ec(problem))
        elif 'n_colors' in problem:
            if random_solutions:
                solution_list.append(get_random_coloring(problem))
            else:
                solution_list.append(get_d_satur_algorithm_solution(problem))
        elif 'clauses' in problem:
            if random_solutions:
                solution_list.append(get_random_m3sat_solution(problem))
            else:
                solution_list.append(get_wsat_solution_m3sat(problem))
        elif 'graph1' in problem and 'graph2' in problem:
            if random_solutions:
                solution_list.append(get_random_sgi_solution(problem))
            else:
                solution_list.append(get_subgraph_iso_solution(problem, 4))
    return solution_list


# Classical heuristic algorithm for TSP:
# Choose one city to start, order the remaining cities regarding the distances and then choose one of the closest
def get_classical_solution_tsp(problem: dict) -> list:
    rng = np_random.default_rng()
    cities_left = {i for i in range(len(problem['cities']))}
    # for city in range(len(problem['cities'])):
    #    print(city)
    #    cities_left.add(city)
    current_city = rng.integers(len(cities_left))
    cities_left.remove(current_city)
    dist_matrix = problem['dist_matrix']
    cities_visited = [current_city]
    while len(cities_left) != 0:
        dist_list = [(dist_matrix[current_city][i], i) for i in cities_left]
        dist_list.sort()
        chosen_spot = rng.binomial(len(cities_left) - 1, 1 / (2 * len(cities_left)))
        current_city = dist_list[chosen_spot][1]
        cities_visited.append(current_city)
        cities_left.remove(current_city)
    return create_solution_from_city_list(cities_visited)


# Build a viable solution from an ordered city list
def create_solution_from_city_list(city_list: list) -> list:
    # solution_list = [0 for _ in range(len(city_list) ** 2)]
    solution_list = np.zeros(len(city_list) ** 2)
    for i in range(len(city_list)):
        city = city_list[i]
        solution_list[city * len(city_list) + i] = 1
    return solution_list


# Use the random differencing heuristic to get a solution for number partitioning
def get_random_differencing_solution(problem: dict) -> list:
    rng = np_random.default_rng()
    numbers = [(number, i) for i, number in enumerate(problem['numbers'])]
    numbers.sort(reverse=True)
    backtracking_list = []
    idx = 0
    while len(numbers) > 1:
        idx -= 1
        biggest = numbers.pop(0)
        next_number = rng.binomial(len(numbers) - 1, 1 / (2 * len(numbers)))
        second = numbers.pop(next_number)
        numbers.append((biggest[0] - second[0], idx))
        backtracking_list.append((biggest, second, (biggest[0] - second[0], idx)))
        numbers.sort(reverse=True)
    sorted_numbers = backtracking(backtracking_list, [[numbers.pop()], []])
    # print('Sorted numbers', sorted_numbers)
    return get_solution_from_sorted_numbers(sorted_numbers)


def backtracking(backtracking_list: list, sorted_numbers: list[list, list]) -> list[list, list]:
    while len(backtracking_list) > 0:
        backtracking_item = backtracking_list.pop()
        idx = backtracking_item[2][1]
        i, j = get_item_ids_from_sorted_numbers(sorted_numbers, idx)
        bigger_number, smaller_number = backtracking_item[0], backtracking_item[1]
        sorted_numbers[i].pop(j)
        sorted_numbers[i].append(bigger_number)
        sorted_numbers[(i + 1) % 2].append(smaller_number)
    return sorted_numbers


def get_item_ids_from_sorted_numbers(sorted_numbers: list[list, list], idx: int) -> tuple[int, int]:
    for i in range(2):
        for j, item in enumerate(sorted_numbers[i]):
            if item[1] == idx:
                return i, j


# Get a solution list from the two lists of indexes
def get_solution_from_sorted_numbers(sorted_numbers: list[list, list]) -> list:
    solution = np.zeros(len(sorted_numbers[0]) + len(sorted_numbers[1]))
    for item in sorted_numbers[0]:
        solution[item[1]] = 1
    return solution


# Simple conditional assignment algorithm as MC heuristic
def get_solution_conditional_assignments_mc(problem: dict) -> list:
    graph = problem['graph']
    node_list = [i for i in range(len(graph.nodes))]
    random.shuffle(node_list)
    cut = [set(), set()]
    for node in node_list:
        count_set0 = 0
        count_set1 = 0
        for edge in [e for e in graph.edges if e[0] == node or e[1] == node]:
            if edge[0] == node:
                other_node = edge[1]
            else:
                other_node = edge[0]
            if other_node in cut[0]:
                count_set0 += 1
            elif other_node in cut[1]:
                count_set1 += 1
        if count_set1 > count_set0:
            cut[0].add(node)
        else:
            cut[1].add(node)
    # print(cut)
    solution = np.zeros(len(node_list))
    for node in cut[1]:
        solution[node] = 1
    return solution


# DLX-algorithm as a heuristic for EC
def get_dlx_solution_ec(problem: dict) -> list:
    subset_matrix = problem['subset_matrix']
    selected_subsets = dlx_algorithm(subset_matrix, [], {i: i for i in range(len(subset_matrix[0]))})
    solution = np.zeros(len(subset_matrix[0]))
    for selected_subset in selected_subsets:
        solution[selected_subset] = 1
    # print(solution)
    return solution


def dlx_algorithm(subset_matrix: list[list], selected_subsets: list, subset_mapping: dict) -> list:
    rng = np_random.default_rng()
    subset = rng.integers(len(subset_matrix[0]))
    selected_subsets.append(subset_mapping[subset])
    new_subset_matrix = reduce_subset_matrix(subset_matrix, subset, subset_mapping)
    if new_subset_matrix and not len(new_subset_matrix[0]) == 0:
        selected_subsets = dlx_algorithm(new_subset_matrix, selected_subsets, subset_mapping)
    return selected_subsets


def reduce_subset_matrix(subset_matrix: list[list], selected_subset: int, subset_mapping: dict) -> list[list]:
    selected_elements = set()
    for element in range(len(subset_matrix)):
        if subset_matrix[element][selected_subset] == 1:
            selected_elements.add(element)
    remaining_elements = [i for i in range(len(subset_matrix)) if i not in selected_elements]
    if not remaining_elements:
        return []
    new_subset_matrix = [[] for _ in remaining_elements]
    # get the subsets still present and add them to the new subset matrix
    for old_subset in range(len(subset_matrix[0])):
        if keep_subset(subset_matrix, old_subset, selected_elements):
            new_id = len(new_subset_matrix[0])
            subset_mapping[new_id] = subset_mapping[old_subset]
            for idx, element in enumerate(remaining_elements):
                new_subset_matrix[idx].append(subset_matrix[element][old_subset])
    return new_subset_matrix


def keep_subset(subset_matrix: list[list], old_subset: int, selected_elements: set) -> bool:
    for element in selected_elements:
        if subset_matrix[element][old_subset] == 1:  # subset contains already selected element
            return False
    return True


# Using a variant of the DSatur-Algorithm to generate a heuristic solution for GC
def get_d_satur_algorithm_solution(problem: dict) -> list:
    graph = problem['graph']
    n_colors = problem['n_colors']
    node_open_color_list = [[n_colors, node, set([i for i in range(n_colors)])] for node in graph.nodes]
    random.shuffle(node_open_color_list)
    node_color_list = [0 for _ in graph.nodes]
    while len(node_open_color_list) > 0:
        node_open_color_list, node_color_list = process_next_urgent_node(node_open_color_list, node_color_list,
                                                                         n_colors, graph)
    return node_color_list_to_solution(node_color_list, n_colors)


def process_next_urgent_node(node_open_color_list: list[tuple[int, int, set[int]]],
                             node_color_list: list[int], n_colors: int, graph: networkx.Graph) \
        -> tuple[list[tuple[int, int, set[int]]], list[int]]:
    current_node = node_open_color_list.pop(0)
    current_node_nr = current_node[1]
    chosen_color = get_color(current_node[2], n_colors)
    new_open_node_color_list = []
    for open_node in node_open_color_list:
        open_colors, open_node_nr, open_colors_dict = open_node
        for edge in graph.edges:  # Search for adjacent nodes
            if edge[0] == current_node_nr and edge[1] == open_node_nr or edge[0] == open_node_nr \
                    and edge[1] == current_node_nr:
                if chosen_color in open_colors_dict:  # If node not already adjacent to chosen color:
                    open_colors -= 1
                    open_colors_dict.remove(chosen_color)
        new_open_node_color_list.append((open_colors, open_node_nr, open_colors_dict))
    node_color_list[current_node_nr] = chosen_color
    new_open_node_color_list.sort()
    return new_open_node_color_list, node_color_list


def node_color_list_to_solution(node_color_list: list, n_colors: int) -> list:
    solution = np.zeros(len(node_color_list * n_colors))
    for node, color in enumerate(node_color_list):
        solution[node * n_colors + color] = 1
    return solution


def get_color(open_colors: set[int], n_colors: int) -> int:
    rng = np_random.default_rng()
    if not open_colors:
        open_colors_list = [i for i in range(n_colors)]
    else:
        open_colors_list = [i for i in open_colors]
    idx = rng.integers(0, len(open_colors_list))
    return open_colors_list[idx]


# Using the simple 1-opt heuristic to generate a possible solution for m3sat
def get_wsat_solution_m3sat(problem: dict, max_walks=5, max_flips=10, flip_prob=.5) -> list[int]:
    rng = np_random.default_rng()
    n_vars = problem['n_vars']
    clause_list = problem['clauses']
    solution = get_random_variable_assignment(n_vars)
    for _ in range(max_walks):
        solution = get_random_variable_assignment(n_vars)
        for _ in range(max_flips):
            if rng.uniform() < flip_prob:
                solution, clauses_satisfied = get_best_flip(solution, clause_list)
            else:
                solution = satisfy_one_random_clause(solution, clause_list)
                clauses_satisfied = check_m3sat_solution(clause_list, solution)
            # print(clauses_satisfied, len(clause_list))
            if clauses_satisfied == len(clause_list):
                return get_qubo_solution_for_m3sat(n_vars, clause_list, solution)
    return get_qubo_solution_for_m3sat(n_vars, clause_list, solution)


def get_best_flip(current_solution: list[bool], clause_list: list[list[tuple[int, bool]]]) -> tuple[list[bool], int]:
    current_best = 0
    current_best_solution = []
    new_solutions = [[val if not idx == i else not val for idx, val in enumerate(current_solution)]
                     for i in range(len(current_solution))]
    random.shuffle(new_solutions)
    for solution in new_solutions:
        solution_quality = check_m3sat_solution(clause_list, solution)
        if solution_quality > current_best:
            current_best_solution = solution
            current_best = solution_quality
    return current_best_solution, current_best


def satisfy_one_random_clause(solution: list[bool], clause_list: list[list[tuple[int, bool]]]):
    unsatisfied_clauses = []
    for clause in clause_list:
        if not check_m3sat_clause(clause, solution):
            unsatisfied_clauses.append(clause)
    # print('Unerfüllte Klauseln', len(unsatisfied_clauses))

    if not unsatisfied_clauses:  # All clauses are satisfied
        return solution

    random.shuffle(unsatisfied_clauses)
    chosen_clause = unsatisfied_clauses[0]
    random.shuffle(chosen_clause)
    chosen_variable, _ = chosen_clause[0]
    solution[chosen_variable] = not solution[chosen_variable]
    return solution


def get_random_variable_assignment(n_vars: int) -> list[bool]:
    rng = np_random.default_rng()
    solution = []
    for i in range(n_vars):
        if rng.uniform() > .5:
            solution.append(True)
        else:
            solution.append(False)
    return solution


def check_m3sat_solution(clause_list: list[list[tuple[int, bool]]], solution: list[bool]) -> int:
    correct_clauses = 0
    for clause in clause_list:
        if check_m3sat_clause(clause, solution):
            correct_clauses += 1
    return correct_clauses


def check_m3sat_clause(clause: list[tuple[int, bool]], solution: list[bool]) -> bool:
    clause_value = False
    for variable, sign in clause:
        clause_value = clause_value or ((solution[variable] and sign) or (not solution[variable] and not sign))
    return clause_value


# Transform the True/False solution into a possible solution from a qubo (first the variables, then each clause)
def get_qubo_solution_for_m3sat(n_vars: int, clause_list: list[list[tuple[int, bool]]], solution: list[bool]) -> list:
    qubo_solution = np.zeros(n_vars + len(clause_list))
    for i in range(n_vars):
        if solution[i]:
            qubo_solution[i] = 1
    for idx, clause in enumerate(clause_list):
        if get_clause_energy(clause, solution) < 0:
            qubo_solution[n_vars + idx] = 1
    return qubo_solution


def get_clause_energy(clause: list[tuple[int, bool]], solution: list[bool]) -> float:
    clause_energy_list = [-0.5 * solution[i] if sign else 0.5 * solution[i] for i, sign in clause]
    return np.sum(clause_energy_list) + get_clause_sum(clause)


def get_clause_sum(clause: list[tuple[int, bool]]) -> float:
    clause_sum = 0.5
    for _, sign in clause:
        if sign:
            clause_sum += .5
        else:
            clause_sum -= .5
    return clause_sum


def node_pairs_to_solution(node_pairs: dict, graph1_order: int, graph2_order: int) -> list[int]:
    solution = np.zeros(graph1_order * graph2_order)
    for node_1 in node_pairs:
        node_2 = node_pairs[node_1]
        solution[node_1 * graph2_order + node_2] = 1
    return solution


def get_subgraph_iso_solution(problem: dict, max_bruteforce_steps: int) -> list[int]:
    graph1 = problem['graph1']
    graph2 = problem['graph2']
    # print(graph1.nodes, graph1.edges)
    # print(graph2.nodes, graph2.edges)
    random_steps = graph1.order() - max_bruteforce_steps
    open_nodes = get_first_suitable_nodes_for_sgi(graph1, graph2)
    found_subgraph = networkx.Graph()
    matched_nodes = {}
    _, _, matched_nodes = get_subgraph_iso_step(graph1, graph2, found_subgraph, matched_nodes, open_nodes, random_steps)
    open_nodes_1, open_nodes_2 = get_open_nodes(matched_nodes, graph1, graph2)
    # print('matched before random', matched_nodes)
    matched_nodes = random_match_open_modes(matched_nodes, open_nodes_1, open_nodes_2)
    # print('Final match ', matched_nodes)
    return node_pairs_to_solution(matched_nodes, graph1.order(), graph2.order())


def get_first_suitable_nodes_for_sgi(graph1: networkx.Graph, graph2: networkx.Graph) -> dict[int, set[int]]:
    suitable_nodes = {}
    node_order_dict_1 = get_node_order_dict_for_graph(graph1)
    node_order_dict_2 = get_node_order_dict_for_graph(graph2)
    for node in graph1.nodes:
        node_order_set = set()
        node_order = node_order_dict_1[node]
        for node_2 in node_order_dict_2:
            if node_order_dict_2[node_2] >= node_order:
                node_order_set.add(node_2)
        if node_order_set:
            suitable_nodes[node] = node_order_set
    return suitable_nodes


def get_node_order_dict_for_graph(graph: networkx.Graph) -> dict:
    node_order_dict = {}
    for node in graph.nodes:
        node_order = 0
        for edge_1, edge_2 in graph.edges:
            if edge_1 == node or edge_2 == node:
                node_order += 1
        node_order_dict[node] = node_order
    return node_order_dict


def get_subgraph_iso_step(graph1: networkx.Graph, graph2: networkx.Graph, found_subgraph: networkx.Graph,
                          matched_nodes: dict, open_nodes: dict[int, set[int]], steps_random: int) \
        -> tuple[networkx.Graph, int, dict[int, int]]:
    # print('matched nodes: ', matched_nodes)
    # print('Found subgraph: ', found_subgraph.nodes, found_subgraph.edges)
    # print('Open nodes ', open_nodes)
    next_node_list = get_nodes_to_continue(graph1, graph2, matched_nodes, found_subgraph, open_nodes)
    # print('next node list ', next_node_list)
    if next_node_list:
        if steps_random > 0:
            return get_random_subgraph_iso_step(graph1, graph2, found_subgraph, matched_nodes, open_nodes,
                                                steps_random, next_node_list)
        else:
            return get_bruteforce_subgraph_iso_step(graph1, graph2, found_subgraph, matched_nodes, open_nodes,
                                                    steps_random, next_node_list)
    return found_subgraph, get_quality_of_subgraph(found_subgraph), matched_nodes


def get_random_subgraph_iso_step(graph1: networkx.Graph, graph2: networkx.Graph, found_subgraph: networkx.Graph,
                                 matched_nodes: dict, open_nodes: dict[int, set[int]], steps_random: int,
                                 next_node_list: list[tuple[int, set[int]]]) -> tuple[networkx.Graph, int,
                                                                                      dict[int, int]]:
    node_1, node_2_set = next_node_list.pop(0)
    random_node2_list = get_random_list_from_set(node_2_set)
    # print('Random step')
    if random_node2_list:
        node_2 = random_node2_list.pop(0)
        matched_nodes[node_1] = node_2
        new_subgraph = extend_subgraph_with_node(graph1, found_subgraph, node_1)
        new_open_nodes = get_new_open_nodes(open_nodes, node_1, node_2)
        steps_random -= 1
        return get_subgraph_iso_step(graph1, graph2, new_subgraph, matched_nodes, new_open_nodes, steps_random)
    else:
        return found_subgraph, get_quality_of_subgraph(found_subgraph), matched_nodes


def get_bruteforce_subgraph_iso_step(graph1: networkx.Graph, graph2: networkx.Graph, found_subgraph: networkx.Graph,
                                     matched_nodes: dict, open_nodes: dict[int, set[int]], steps_random: int,
                                     next_node_list: list[tuple[int, set[int]]]) -> tuple[networkx.Graph, int,
                                                                                          dict[int, int]]:
    best_subgraph = found_subgraph
    best_quality = get_quality_of_subgraph(found_subgraph)
    best_matched_nodes = matched_nodes
    # print('Bruteforce step')
    for node_1, node_2_set in next_node_list:
        for node_2 in node_2_set:
            subgraph, quality, matched_return = get_subgraph_iso_step(graph1, graph2,
                                                                      extend_subgraph_with_node(graph1, found_subgraph,
                                                                                                node_1),
                                                                      create_new_matched_nodes_dict(matched_nodes,
                                                                                                    node_1, node_2),
                                                                      get_new_open_nodes(open_nodes, node_1, node_2),
                                                                      steps_random)
            if quality > best_quality:
                best_subgraph = subgraph
                best_quality = quality
                best_matched_nodes = matched_return
                if quality == len(graph1.edges):
                    return best_subgraph, best_quality, best_matched_nodes
    return best_subgraph, best_quality, best_matched_nodes


def get_nodes_to_continue(graph1: networkx.Graph, graph2: networkx.Graph, matched_nodes: dict,
                          found_subgraph: networkx.Graph, open_nodes: dict[int, set[int]]) -> \
        list[tuple[int, set[int]]]:
    open_node_list = []
    for node in open_nodes:
        suitable_nodes = get_suitable_nodes(graph1, graph2, matched_nodes, found_subgraph, node, open_nodes[node])
        # print('Node ', node, suitable_nodes)
        if suitable_nodes:
            open_node_list.append((node, suitable_nodes))
    random.shuffle(open_node_list)
    return open_node_list


def get_suitable_nodes(graph1: networkx.Graph, graph2: networkx.Graph, matched_nodes: dict[int, int],
                       found_subgraph: networkx.Graph, choosen_node: int, open_nodes_2: set[int]) -> set[int]:
    possible_nodes = set()
    for open_node in open_nodes_2:
        possible_nodes.add(open_node)
        for edge_1, edge_2 in graph1.edges:
            if (edge_1 == choosen_node and edge_2 in found_subgraph.nodes and
                not ((open_node, matched_nodes[edge_2]) in graph2.edges or
                     (matched_nodes[edge_2], open_node) in graph2.edges)) \
                    or (edge_2 == choosen_node and edge_1 in found_subgraph.nodes and
                        not ((open_node, matched_nodes[edge_1]) in graph2.edges
                             or (matched_nodes[edge_1], open_node) in graph2.edges)):
                possible_nodes.remove(open_node)
                break
    return possible_nodes


def extend_subgraph_with_node(graph1: networkx.Graph, subgraph: networkx.Graph, node: int) -> networkx.Graph:
    new_subgraph = networkx.Graph()
    for node_old in subgraph.nodes:
        new_subgraph.add_node(node_old)
    new_subgraph.add_node(node)
    for edge_1, edge_2 in graph1.edges:
        if edge_1 in new_subgraph.nodes and edge_2 in new_subgraph.nodes:
            new_subgraph.add_edge(edge_1, edge_2)
    return new_subgraph


def create_new_matched_nodes_dict(matched_nodes: dict[int, int], node1: int, node2: int) -> dict[int, int]:
    new_dict = {}
    for node in matched_nodes:
        new_dict[node] = matched_nodes[node]
    new_dict[node1] = node2
    return new_dict


def get_new_open_nodes(open_nodes: dict[int, set[int]], node_1: int, node_2: int) -> dict[int, set[int]]:
    new_dict = {}
    for node in open_nodes:
        if not node == node_1:
            new_dict[node] = set()
            for node_2_ in open_nodes[node]:
                if node_2_ != node_2:
                    new_dict[node].add(node_2_)
    return new_dict


def get_quality_of_subgraph(found_subgraph: networkx.Graph) -> int:
    return len(found_subgraph.edges)


def get_random_list_from_set(input_set: set) -> any:
    random_list = [a for a in input_set]
    random.shuffle(random_list)
    return random_list


def get_open_nodes(matched_nodes: dict, graph1: networkx.Graph, graph2: networkx.Graph) -> tuple[list[int], list[int]]:
    used_nodes1 = matched_nodes.keys()
    used_nodes2 = matched_nodes.values()
    open_nodes_1 = []
    for node in graph1.nodes:
        if node not in used_nodes1:
            open_nodes_1.append(node)
    open_nodes_2 = []
    for node in graph2.nodes:
        if node not in used_nodes2:
            open_nodes_2.append(node)
    return open_nodes_1, open_nodes_2


def random_match_open_modes(matched_nodes: dict, open_nodes_1: list[int], open_nodes_2: list[int]) -> dict:
    random.shuffle(open_nodes_2)
    for node in open_nodes_1:
        node_2 = open_nodes_2.pop(0)
        matched_nodes[node] = node_2
    return matched_nodes


# # # # # # #                                                     # # # # # # #
#                          Random assignment methods                          #
# # # # # # #                                                     # # # # # # #

def get_random_city_order_solution(problem: dict) -> list:
    cities = [i for i in range(len(problem['cities']))]
    random.shuffle(cities)
    return create_solution_from_city_list(cities)


def get_random_solution_assignment(n: int) -> list:
    rng = np_random.default_rng()
    solution = np.zeros(n)
    p = rng.uniform()
    for i in range(n):
        if rng.uniform() < p:
            solution[i] = 1
    return solution


def get_random_coloring(problem: dict) -> list:
    rng = np_random.default_rng()
    graph = problem['graph']
    n_colors = problem['n_colors']
    node_count = len(graph.nodes)
    solution = np.zeros(node_count * n_colors)
    for node in range(node_count):
        random_color = rng.integers(0, n_colors)
        solution[node * n_colors + random_color] = 1
    return solution


def get_random_m3sat_solution(problem: dict) -> list[int]:
    n_vars, clauses = problem['n_vars'], problem['clauses']
    random_solution = get_random_variable_assignment(n_vars)
    return get_qubo_solution_for_m3sat(n_vars, clauses, random_solution)


def get_random_sgi_solution(problem: dict) -> list[int]:
    graph1 = problem['graph1']
    graph2 = problem['graph2']
    node_dict = random_match_open_modes({}, [i for i in graph1.nodes], [i for i in graph2.nodes])
    return node_pairs_to_solution(node_dict, graph1.order(), graph2.order())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                               Old and outdated Functions                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
        solution_quality, best_approx_solution, true_approx_percent, *_ = get_quality_of_approxed_qubo(
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

# problem = {'numbers': [120, 75, 74, 73, 71, 60, 56, 54, 46, 15]}
# for n in range(10):
# print(get_random_differencing_solution(problem))
# print(get_random_solution_assignment(10))

# problem = {'subset_matrix': [[0, 0, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0, 0],
#                             [1, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 1, 0, 0]]}

# print(get_dlx_solution_ec(problem))
