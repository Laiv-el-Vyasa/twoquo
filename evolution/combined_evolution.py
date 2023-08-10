import os
import sys

import numpy as np
import torch
import time

from pipeline import pipeline_run
from visualisation import qubo_heatmap, visualize_evol_results, plot_average_fitness, visualize_evol_results_models
from pygad.torchga import torchga
from model_data import evaluation_models, model_dict

import pygad.torchga

cuda = torch.device('cuda')

from evolution.evolution_util import get_training_dataset, get_fitness_value, apply_approximation_to_qubo, \
    get_quality_of_approxed_qubo, get_qubo_approx_mask, aggregate_saved_problems, solver, \
    check_model_config_fit, get_diagonal_of_qubo, cfg, get_tensor_of_structure, linearize_qubo, \
    check_pipeline_necessity, check_solution, get_mean_of_qubo_line


# *---------------------------------------------------------------------------------- * #
#                                                                                       #
#                       FILES DEPRECATED - CHECK NEW EVOLUTION                          #
#                                                                                       #
# *---------------------------------------------------------------------------------- * #

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

run_bool = False
restart_bool = False
extend_model = False
model_to_extend = 'combined_evolution_MC_24_uwu_1_05_10_01_005'
test_case_study = False
plot_evol_results = True
compare_types = True

qubo_size = cfg['pipeline']['problems']['qubo_size']
problem = cfg['pipeline']['problems']['problems'][0]
problem_count = cfg['pipeline']['problems']['n_problems']
fitness_problem_percent = .1
evolution_type = 'combined'
evolution_file = f"{evolution_type}_evolution"
model_name = '_uwu'
qubo_entries = int((qubo_size * (qubo_size + 1)) / 2)
population = 100
a = 1
b = .5
c = 10
d = .1
fitness_parameters = (a, b, c, d)
min_approx = 0.05

# node_model = CombinedNodeFeatures(node_feature_number)
if evolution_type == 'combined':
    node_model = model_dict[f'model{model_name}'][0]
    node_model_normalized = model_dict[f'model{model_name}_normalized'][0]
    edge_model = model_dict[f'model{model_name}'][1]
    # edge_model = CombinedEdgeDecision(node_feature_number)

    torch_ga_node = pygad.torchga.TorchGA(model=node_model, num_solutions=population)
    torch_ga_edge = pygad.torchga.TorchGA(model=edge_model, num_solutions=population)
else:
    model = model_dict[evolution_type][f'model{model_name}']
    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=population)

avg_fitness_list = []
avg_fitness_generation = []
best_fitness = 0

complete_fitness_time = []

global_node_model_dict = {}


def get_node_model_and_features(problem, qubo):
    global node_model, node_model_normalized
    return_node_model = node_model
    return_node_features = get_diagonal_of_qubo(qubo)
    if 'graph' in problem and 'tsp' in problem:
        # print('Normalized model choosen!')
        return_node_model = node_model_normalized
        # return_node_features = get_mean_of_qubo_line(qubo)
    return return_node_model, return_node_features


def get_linarized_approx(evo_type, fitness_bool):
    # linearized_return = []
    if evo_type == 'combined':
        linearized_return = get_linearized_approx_combined(fitness_bool)
    elif evo_type == 'gcn':
        linearized_return = get_linearized_approx_gcn(fitness_bool)
    else:
        linearized_return = get_linearized_approx_simple(fitness_bool)
    return linearized_return


def get_linearized_approx_combined(fitness_bool):
    global edge_model, problem_count, fitness_problem_percent
    n_problems = problem_count
    if fitness_bool:
        n_problems = int(fitness_problem_percent * problem_count)
    _, qubos, min_energy, solutions_list, problem_list, edge_index_list, edge_weight_list = \
        get_training_dataset(n_problems, include_loops=True)
    linearized_approx = []
    problem_time_list = []
    for qubo, edge_index, edge_weight, problem in zip(qubos, edge_index_list, edge_weight_list, problem_list):
        # print('QUBO:', qubo)
        # print('Problem approxxing: ', problem)
        problem_time = time.time()
        use_node_model, node_features = get_node_model_and_features(problem, qubo)
        # print(node_features)
        node_features = use_node_model.forward(get_tensor_of_structure(node_features),
                                               get_tensor_of_structure(edge_index).long(),
                                               get_tensor_of_structure(edge_weight)).detach()
        # print('Node features', node_features)
        approx_mask = np.ones((len(qubo), len(qubo)))
        node_mean_tensor_list = []
        for edge_0, edge_1 in zip(edge_index[0], edge_index[1]):
            node_features_0 = np.array(node_features[edge_0].numpy())
            node_features_1 = np.array(node_features[edge_1].numpy())
            # print(node_features_0, node_features_1, edge_0, edge_1)
            # print(np.mean(node_features_0, node_features_1))
            # node_mean_tensor = np.mean([node_features_0, node_features_1], axis=0)
            node_mean_tensor_list.append(np.mean([node_features_0, node_features_1], axis=0))
        # print(node_mean_tensor_list)

        edge_descision_list = edge_model.forward(get_tensor_of_structure(node_mean_tensor_list)).detach()
        # edge_descision = edge_model.forward(get_tensor_of_structure(node_mean_tensor)).detach()
        for idx, edge_descision in enumerate(edge_descision_list):
            # print(edge_descision.cpu().detach())
            if edge_descision.detach() <= 0:
                approx_mask[edge_index[0][idx]][edge_index[1][idx]] = 0
                # approx_mask[edge_0][edge_1] = 0
        # print(approx_mask)
        # print(np.mean(node_features_0, node_features_1))
        linearized_approx.append(linearize_qubo(approx_mask))
        # print('Problem approxxing time: ', time.time() - problem_time)
        problem_time_list.append(time.time() - problem_time)

    # print('Max time aproxxing problem: ', problem_list[np.argmax(problem_time_list)])
    # print('Max time aproxxing: ', np.max(problem_time_list))
    return linearized_approx, qubos, min_energy, solutions_list, problem_list


def get_linearized_approx_gcn(fitness_bool):
    global model, model_name, problem_count, fitness_problem_percent
    n_problems = problem_count
    if fitness_bool:
        n_problems = int(fitness_problem_percent * problem_count)
    include_loops = not (model_name == '_diag' or model_name == '_deep')
    _, qubos, min_energy, solutions_list, problem_list, edge_index_list, edge_weight_list = get_training_dataset(
        n_problems,
        include_loops=include_loops)
    linearized_approx = []
    for qubo, edge_index, edge_weight in zip(qubos, edge_index_list, edge_weight_list):
        # print('QUBO:', qubo)
        if model_name == '_diag':
            node_features = get_diagonal_of_qubo(qubo)
        elif model_name == '_deep':
            node_features = qubo
            edge_weight = []
        else:
            node_features = np.identity(len(qubo))
        approx_mask = model.forward(get_tensor_of_structure(node_features),
                                    get_tensor_of_structure(edge_index).long(),
                                    get_tensor_of_structure(edge_weight))
        # print('APPROX', approx_mask)
        linearized_approx.append(linearize_qubo(approx_mask.detach()))
    return linearized_approx, qubos, min_energy, solutions_list, problem_list


def get_linearized_approx_simple(fitness_bool):
    global model, problem_count, fitness_problem_percent
    n_problems = problem_count
    if fitness_bool:
        n_problems = int(fitness_problem_percent * problem_count)
    linearized_qubos, qubos, min_energy, solution_lists, problem_list, *_ = get_training_dataset(n_problems)
    linearized_approx = model(linearized_qubos).detach()
    return linearized_approx, qubos, min_energy, solutions_list, problem_list


def fitness_func(solution, solution_idx):
    global torch_ga, node_model, node_model_normalized, edge_model, avg_fitness_list, avg_fitness_generation, \
        min_approx, node_edge_cutoff, best_fitness, complete_fitness_time, evolution_type

    start = time.time()
    if evolution_type == 'combined':
        node_model = model_dict[f'model{model_name}'][0]
        node_model_normalized = model_dict[f'model{model_name}_normalized'][0]
        edge_model = model_dict[f'model{model_name}'][1]
        # print(node_model)
        # print(edge_model)

        model_weights_dict_node = torchga.model_weights_as_dict(model=node_model,
                                                                weights_vector=solution[:node_edge_cutoff])
        node_model.load_state_dict(model_weights_dict_node)
        node_model_normalized.load_state_dict(model_weights_dict_node)

        model_weights_dict_edge = torchga.model_weights_as_dict(model=edge_model,
                                                                weights_vector=solution[node_edge_cutoff:])
        edge_model.load_state_dict(model_weights_dict_edge)
        # print('In Fitness Func: ', node_model)
    else:
        model = model_dict[evolution_type][f'model{model_name}']
        model_weights_dict = torchga.model_weights_as_dict(model=model, weights_vector=solution)
        model.load_state_dict(model_weights_dict)
        # print('In Fitness Func: ', model)

    linearized_approx, qubos, min_energy, solutions_list, problems = get_linarized_approx(evolution_type,
                                                                                          fitness_bool=True)
    # print('Lin approx: ', linearized_approx)
    # print('Len Fitness', len(qubos))
    solution_fitness = get_fitness_value(linearized_approx, qubos, min_energy, solutions_list, fitness_parameters,
                                         problems, min_approx)
    print(f'Solution {solution_idx}: {solution_fitness}')
    if solution_fitness > best_fitness:
        best_fitness = solution_fitness
    avg_fitness_generation.append(solution_fitness)
    time_spent = time.time() - start
    # print('Recent time: ', time_spent)
    complete_fitness_time.append(time_spent)
    # print('Avg time so far: ', np.mean(complete_fitness_time))
    return solution_fitness


def callback_generation(ga_instance):
    global avg_fitness_generation, avg_fitness_list, best_fitness, complete_fitness_time
    print("Generation   = {generation}".format(generation=ga_instance.generations_completed))

    # print("Fitness      = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Fitness      = {fitness}".format(fitness=best_fitness))
    # print(avg_fitness_generation)
    avg_fitness = np.mean(avg_fitness_generation)
    avg_fitness_list.append(avg_fitness)
    avg_fitness_generation = []
    print("Avg. Fitness = {fitness}".format(fitness=avg_fitness))
    print("Avg. Runtime = {time}".format(time=np.mean(complete_fitness_time)))
    complete_fitness_time = []


num_generations = 50
num_parents_mating = int(population * .2)
if evolution_type == 'combined':
    initial_population_node = torch_ga_node.population_weights
    node_edge_cutoff = len(initial_population_node[0])
    initial_population_edge = torch_ga_edge.population_weights
    initial_population = np.append(initial_population_node, initial_population_edge, axis=-1)
    print(
        f'{node_edge_cutoff} + {len(initial_population_edge[0])} = {node_edge_cutoff + len(initial_population_edge[0])}, '
        f'{len(initial_population[0])}')
else:
    initial_population = torch_ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations,
                       # parent_selection_type='rws',
                       keep_elitism=5,
                       # crossover_type='scattered',
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

if extend_model:
    if not restart_bool:
        sys.exit('Extend nur mit Restart möglich')
    ga_instance_loaded = pygad.load(f'{model_to_extend}')
    print('Loaded generations completed: ', ga_instance_loaded.generations_completed)
    initial_population = ga_instance_loaded.population
    print(initial_population)
    ga_instance = pygad.GA(num_generations=num_generations,
                           # parent_selection_type='rws',
                           keep_elitism=5,
                           # crossover_type='scattered',
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           on_generation=callback_generation)

if run_bool:
    print(f'{evolution_file}_{problem}_{qubo_size}{model_name}')
    if not restart_bool:
        ga_instance = pygad.load(f'{evolution_file}_{problem}_{qubo_size}{model_name}')
    print('Run generations completed: ', ga_instance.generations_completed)
    ga_instance.run()
    ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness")
    plot_average_fitness(avg_fitness_list)
    ga_instance.save(f'{evolution_file}_{problem}_{qubo_size}{model_name}')

test_model = 'combined_evolution_NP_48_uwu_1_05_10_02_01'
test_cases = 10
if test_case_study and evaluation_models[test_model] and \
        check_model_config_fit(test_model, evaluation_models[test_model]['independence']):
    fitness_parameters = evaluation_models[test_model]['fitness_params']
    min_approx = evaluation_models[test_model]['min_approx']
    evolution_type = evaluation_models[test_model]['evolution_type']
    loaded_ga_instance = pygad.load(f'{test_model}')
    model_name = evaluation_models[test_model]["model_name"]
    print(test_model, model_name, evolution_type, min_approx)
    # model = model_dict[f'model{evaluation_models[model_descr]["model_name"]}']
    if evolution_type == 'combined':
        node_model = model_dict['combined'][f'model{model_name}'][0]
        node_edge_cutoff = sum(p.numel() for p in node_model.parameters() if p.requires_grad)
        print(node_edge_cutoff)
        edge_model = model_dict['combined'][f'model{model_name}'][1]

        best_solution_tuple = loaded_ga_instance.best_solution()
        # print(best_solution_tuple)
        best_solution = best_solution_tuple[0]
        model_weights_dict_node = torchga.model_weights_as_dict(model=node_model,
                                                                weights_vector=best_solution[:node_edge_cutoff])
        node_model.load_state_dict(model_weights_dict_node)

        model_weights_dict_edge = torchga.model_weights_as_dict(model=edge_model,
                                                                weights_vector=best_solution[node_edge_cutoff:])
        edge_model.load_state_dict(model_weights_dict_edge)
        print('After evaluation: ', node_model)
    else:
        model = model_dict[evolution_type][f'model{model_name}']
        best_solution_tuple = loaded_ga_instance.best_solution()
        # print(best_solution_tuple)
        best_solution = best_solution_tuple[0]
        model_weights_dict = torchga.model_weights_as_dict(model=model, weights_vector=best_solution)
        model.load_state_dict(model_weights_dict)
        print('After evaluation: ', model)

    linearized_approx, qubos, min_energy, solutions_list, problem_list = get_linarized_approx(evolution_type,
                                                                                              fitness_bool=False)
    print('Len eval', len(qubos))

    solution_quality_list = [[], [], []]
    approx_percent = []
    fitness_list = []

    for idx, (lin_approx, qubo, energy, solutions, problem) in enumerate(
            zip(linearized_approx, qubos, min_energy, solutions_list, problem_list)):
        approxed_qubo, _ = apply_approximation_to_qubo(lin_approx, qubo)
        (solution_quality, best_approx_solution), best_approx_solutions, true_approx, true_approx_percent = \
            get_quality_of_approxed_qubo(lin_approx, qubo, solutions)
        # print(f'True approx: {true_approx}')
        solution_check_value = check_solution(solutions, best_approx_solutions, problem)
        if idx < test_cases:
            print(f'Testcase {idx + 1}')
            print(f'Quality of approx: {solution_quality}')
            print(f'Number of approx: {true_approx}')
            print(f'Problem: {problem}')
            print(f'True solution: {solutions}, {energy}')
            print(f'Best approxed solution: {best_approx_solutions}')
            print(f'Summary: Quality: {solution_quality}, '
                  f'Solution value: {solution_check_value}')
            qubo_heatmap(qubo)
            qubo_heatmap(get_qubo_approx_mask(lin_approx, qubo))
            qubo_heatmap(approxed_qubo)
        solution_quality_list[0].append(solution_quality)
        solution_quality_list[1].append(np.floor(1 - solution_quality))
        solution_quality_list[2].append(solution_check_value)
        approx_percent.append(true_approx_percent)

    print(f'Model Fitness: Approx percent: {np.mean(approx_percent)}, '
          f'Solution quality: {np.mean(solution_quality_list[0])}, '
          f'True solution percentage: {np.mean(solution_quality_list[1])}, '
          f'Problem solved percentage: {np.mean(solution_quality_list[2])}')

evol_data = []

if plot_evol_results:
    if False: #check_pipeline_necessity(problem_count):
        approx_single_entries = True
        if qubo_size > 24:
            approx_single_entries = False
        pipeline_db = pipeline_run(cfg, True, approx_single_entries, False, True, False,
                                   check_pipeline_necessity_bool=True, approximation_steps=99)
    for model_descr in evaluation_models:
        fitting_model = True#check_model_config_fit(model_descr, evaluation_models[model_descr]['independence'])
        if fitting_model and evaluation_models[model_descr]['display']:
            fitness_parameters = evaluation_models[model_descr]['fitness_params']
            min_approx = evaluation_models[model_descr]['min_approx']
            evolution_type = evaluation_models[model_descr]['evolution_type']
            loaded_ga_instance = pygad.load(f'{model_descr}')
            model_name = evaluation_models[model_descr]["model_name"]
            print(model_descr, model_name, evolution_type, min_approx)
            # model = model_dict[f'model{evaluation_models[model_descr]["model_name"]}']
            if evolution_type == 'combined':
                node_model = model_dict['combined'][f'model{model_name}'][0]
                node_edge_cutoff = sum(p.numel() for p in node_model.parameters() if p.requires_grad)
                print(node_edge_cutoff)
                edge_model = model_dict['combined'][f'model{model_name}'][1]
                try:
                    model_weights_dict_node = torch.load(f'best_model_{evolution_type}_{problem}_{qubo_size}_node')
                    model_weights_dict_edge = torch.load(f'best_model_{evolution_type}_{problem}_{qubo_size}_edge')
                    print('Stored models loaded')
                except FileNotFoundError:
                    print(loaded_ga_instance.population)
                    np.save(f'saved_population_{evolution_type}_{problem}_{qubo_size}', loaded_ga_instance.population)
                    best_solution_tuple = loaded_ga_instance.best_solution()
                    # print(best_solution_tuple)
                    best_solution = best_solution_tuple[0]
                    model_weights_dict_node = torchga.model_weights_as_dict(
                        model=node_model,
                        weights_vector=best_solution[:node_edge_cutoff]
                    )
                    torch.save(model_weights_dict_node,
                               f'best_model_{evolution_type}_{problem}_{qubo_size}_node')
                    model_weights_dict_edge = torchga.model_weights_as_dict(
                        model=edge_model,
                        weights_vector=best_solution[node_edge_cutoff:]
                    )
                    torch.save(
                        model_weights_dict_edge,
                        f'best_model_{evolution_type}_{problem}_{qubo_size}_edge'
                    )
                node_model.load_state_dict(model_weights_dict_node)
                edge_model.load_state_dict(model_weights_dict_edge)
                print('After evaluation: ', node_model)
            else:
                model = model_dict[evolution_type][f'model{model_name}']
                best_solution_tuple = loaded_ga_instance.best_solution()
                # print(best_solution_tuple)
                best_solution = best_solution_tuple[0]
                model_weights_dict = torchga.model_weights_as_dict(model=model, weights_vector=best_solution)
                model.load_state_dict(model_weights_dict)
                print('After evaluation: ', model)

            linearized_approx, qubos, min_energy, solutions_list, problem_list = \
                get_linarized_approx(evolution_type, fitness_bool=False)
            print('Len eval', len(qubos))

            solution_quality_list = [[], [], []]
            approx_percent = []
            fitness_list = []

            for idx, (lin_approx, qubo, solutions, prob) in enumerate(
                    zip(linearized_approx, qubos, solutions_list, problem_list)):
                (solution_quality, best_solution), appros_solutions, true_approx, true_approx_percent = \
                    get_quality_of_approxed_qubo(lin_approx, qubo, solutions)
                approx_percent.append(true_approx_percent)
                print(solution_quality)
                solution_quality_list[0].append(solution_quality)
                solution_quality_list[1].append(np.floor(1 - solution_quality))
                solution_quality_list[2].append(check_solution(solutions, appros_solutions, prob))

            evol_results = []
            for metric, results in enumerate(solution_quality_list):
                evol_results.append((approx_percent, np.mean(results)))
            evol_data.append((evol_results, evaluation_models[model_descr]['name'],
                              evaluation_models[model_descr]['evolution_type']))

    if evol_data:
        aggregated_problems, approx_percent_array = aggregate_saved_problems(pipeline_db, true_approx=True)
        if compare_types:
            visualize_evol_results_models(aggregated_problems,
                                          approx_percent_array,
                                          evol_data, solver, qubo_size, problem, boxplot=True)
            visualize_evol_results_models(aggregated_problems,
                                          approx_percent_array,
                                          evol_data, solver, qubo_size, problem, metric=2)
        else:
            visualize_evol_results(aggregated_problems,
                                   approx_percent_array,
                                   evol_data, solver, qubo_size)
