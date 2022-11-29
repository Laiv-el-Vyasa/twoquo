import os

import numpy as np

from visualisation import qubo_heatmap, visualize_evol_results, plot_average_fitness
from pygad.torchga import torchga
from networks import CombinedNodeFeatures, CombinedEdgeDecision, CombinedNodeFeaturesNonLin, CombinedEdgeDecisionNonLin

import pygad.torchga

from evolution.evolution_util import get_training_dataset, get_fitness_value, apply_approximation_to_qubo, \
    get_quality_of_approxed_qubo, get_qubo_approx_mask, aggregate_saved_problems, solver, \
    check_model_config_fit, get_diagonal_of_qubo, cfg, get_tensor_of_structure, linearize_qubo

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

run_bool = True
restart_bool = True
test_case_study = False
plot_evol_results = False


#cfg = load_cfg(cfg_id='test_evol_mc')
qubo_size = cfg['pipeline']['problems']['qubo_size']
problem = cfg['pipeline']['problems']['problems'][0]
evolution_file = f"combined_evolution"
model_name = '_nonlin'
qubo_entries = int((qubo_size * (qubo_size + 1)) / 2)
population = 100
a = 1
b = .5
c = 10
d = .1
fitness_parameters = (a, b, c, d)
min_approx = 0.05
node_feature_number = 8


evaluation_models = {
    '_MC_8_1_05_10_01_005':
        {'name': 'combined model',
         'fitness_params': (1, .5, 10, .1),
         'min_approx': 0.05,
         'combined': True,
         'model_name': ''
         },
    '_MC_24_1_05_10_01_005':
        {'name': 'combined model',
         'fitness_params': (1, .5, 10, .1),
         'min_approx': 0.05,
         'combined': True,
         'model_name': ''
         },
    '_MC_24_nonlin_1_05_10_01_005':
        {'name': 'combined model, non-linear',
         'fitness_params': (1, .5, 10, .1),
         'min_approx': 0.05,
         'combined': True,
         'model_name': '_nonlin'
         }
}

model_dict = {
    'model': [
        CombinedNodeFeatures(node_feature_number),
        CombinedEdgeDecision(node_feature_number)
    ],
    'model_nonlin': [
        CombinedNodeFeaturesNonLin(node_feature_number),
        CombinedEdgeDecisionNonLin(node_feature_number)
    ]
}


#node_model = CombinedNodeFeatures(node_feature_number)
node_model = model_dict[f'model{model_name}'][0]
edge_model = model_dict[f'model{model_name}'][1]
#edge_model = CombinedEdgeDecision(node_feature_number)

torch_ga_node = pygad.torchga.TorchGA(model=node_model, num_solutions=population)
torch_ga_edge = pygad.torchga.TorchGA(model=edge_model, num_solutions=population)

avg_fitness_list = []
avg_fitness_generation = []


def fitness_func(solution, solution_idx):
    global torch_ga, node_model, edge_model, avg_fitness_list, avg_fitness_generation, min_approx, node_edge_cutoff

    node_model = model_dict[f'model{model_name}'][0]
    edge_model = model_dict[f'model{model_name}'][1]
    print(node_model)

    model_weights_dict_node = torchga.model_weights_as_dict(model=node_model,
                                                            weights_vector=solution[:node_edge_cutoff])
    node_model.load_state_dict(model_weights_dict_node)

    model_weights_dict_edge = torchga.model_weights_as_dict(model=edge_model,
                                                            weights_vector=solution[node_edge_cutoff:])
    edge_model.load_state_dict(model_weights_dict_edge)

    _, qubos, min_energy, _, _, edge_index_list, edge_weight_list = get_training_dataset(include_loops=True)
    linearized_approx = []
    for qubo, edge_index, edge_weight in zip(qubos, edge_index_list, edge_weight_list):
        #print('QUBO:', qubo)
        node_features = get_diagonal_of_qubo(qubo)
        node_features = node_model.forward(get_tensor_of_structure(node_features),
                                    get_tensor_of_structure(edge_index).long(),
                                    get_tensor_of_structure(edge_weight)).detach()
        #print(node_features)
        approx_mask = np.ones((qubo_size, qubo_size))
        for edge_0, edge_1 in zip(edge_index[0], edge_index[1]):
            node_features_0 = np.array(node_features[edge_0].numpy())
            node_features_1 = np.array(node_features[edge_1].numpy())
            #print(node_features_0, node_features_1, edge_0, edge_1)
            #print(np.mean(node_features_0, node_features_1))
            node_mean_tensor = get_tensor_of_structure(np.mean([node_features_0, node_features_1], axis=0))
            #print(node_mean_tensor)
            edge_descision = edge_model.forward(node_mean_tensor).detach()
            if edge_descision <= 0:
                approx_mask[edge_0][edge_1] = 0
        #print(approx_mask)
        #print(np.mean(node_features_0, node_features_1))
        linearized_approx.append(linearize_qubo(approx_mask))

    solution_fitness = get_fitness_value(linearized_approx, qubos, min_energy, fitness_parameters, min_approx)
    print(f'Solution {solution_idx}: {solution_fitness}')
    avg_fitness_generation.append(solution_fitness)

    return solution_fitness


def callback_generation(ga_instance):
    global avg_fitness_generation, avg_fitness_list
    print("Generation   = {generation}".format(generation=ga_instance.generations_completed))
    avg_fitness_generation = []
    print("Fitness      = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    #print(avg_fitness_generation)
    avg_fitness = np.mean(avg_fitness_generation)
    avg_fitness_list.append(avg_fitness)
    print("Avg. Fitness = {fitness}".format(fitness=avg_fitness))


num_generations = 100
num_parents_mating = int(population * .2)
initial_population_node = torch_ga_node.population_weights
node_edge_cutoff = len(initial_population_node[0])
initial_population_edge = torch_ga_edge.population_weights
initial_population = np.append(initial_population_node, initial_population_edge, axis=-1)
print(f'{node_edge_cutoff} + {len(initial_population_edge[0])} = {node_edge_cutoff + len(initial_population_edge[0])}, '
      f'{len(initial_population[0])}')

ga_instance = pygad.GA(num_generations=num_generations,
                       #parent_selection_type='rws',
                       keep_elitism=5,
                       #crossover_type='scattered',
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

if run_bool:
    if not restart_bool:
        ga_instance = pygad.load(f'{evolution_file}_{problem}_{qubo_size}{model_name}')
    ga_instance.run()
    ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness")
    plot_average_fitness(avg_fitness_list)
    ga_instance.save(f'{evolution_file}_{problem}_{qubo_size}{model_name}')


test_model = '_MC_24_nonlin_1_05_10_01_005'
test_cases = 10
if test_case_study:
    fitness_parameters = evaluation_models[test_model]['fitness_params']
    min_approx = evaluation_models[test_model]['min_approx']
    model_name = evaluation_models[test_model]["model_name"]
    node_model = model_dict[f'model{model_name}'][0]
    edge_model = model_dict[f'model{model_name}'][1]
    #model = model_dict[f'model{evaluation_models[test_model]["model_name"]}']
    loaded_ga_instance = pygad.load(evolution_file + test_model)
    #loaded_ga_instance = pygad.load(evolution_file)

    best_solution_tuple = loaded_ga_instance.best_solution()
    best_solution = best_solution_tuple[0]
    model_weights_dict_node = torchga.model_weights_as_dict(model=node_model,
                                                            weights_vector=best_solution[:node_edge_cutoff])
    node_model.load_state_dict(model_weights_dict_node)

    model_weights_dict_edge = torchga.model_weights_as_dict(model=edge_model,
                                                            weights_vector=best_solution[node_edge_cutoff:])
    edge_model.load_state_dict(model_weights_dict_edge)


    linearized_qubos, qubos, min_energy, solution_list, problem_list, \
    edge_index_list, edge_weight_list = get_training_dataset(include_loops=True)

    solution_quality_list = [[], []]
    approx_quality_list = []
    fitness_list = []
    for idx, (linarized_qubo, qubo, energy, solution, problem, edge_index, edge_weight) in enumerate(
            zip(linearized_qubos, qubos, min_energy, solution_list,
                problem_list, edge_index_list, edge_weight_list)):

        node_features = get_diagonal_of_qubo(qubo)
        node_features = node_model.forward(get_tensor_of_structure(node_features),
                                           get_tensor_of_structure(edge_index).long(),
                                           get_tensor_of_structure(edge_weight)).detach()
        approx_mask = np.ones((qubo_size, qubo_size))
        for edge_0, edge_1 in zip(edge_index[0], edge_index[1]):
            node_features_0 = np.array(node_features[edge_0].numpy())
            node_features_1 = np.array(node_features[edge_1].numpy())
            node_mean_tensor = get_tensor_of_structure(np.mean([node_features_0, node_features_1], axis=0))
            edge_descision = edge_model.forward(node_mean_tensor).detach()
            if edge_descision <= 0:
                approx_mask[edge_0][edge_1] = 0
        lin_approx = linearize_qubo(approx_mask)

        approxed_qubo, true_approx = apply_approximation_to_qubo(lin_approx, qubo)
        solution_quality, _ = get_quality_of_approxed_qubo(lin_approx, qubo, energy, print_solutions=True)
        #print(f'True approx: {true_approx}')
        if idx < test_cases:
            print(f'Testcase {idx + 1}')
            print(f'Quality of approx: {solution_quality}')
            print(f'Number of approx: {true_approx}')
            print(f'True solution: {solution}, {energy}')
            print(f'Problem: {problem}')
            qubo_heatmap(qubo)
            qubo_heatmap(get_qubo_approx_mask(lin_approx))
            qubo_heatmap(approxed_qubo)
        solution_quality_list[0].append(solution_quality)
        solution_quality_list[1].append(np.floor(1 - solution_quality))
        approx_quality_list.append(true_approx)

    print(f'Model Fitness: Approx percent: {np.mean(approx_quality_list) / qubo_entries}, '
          f'Solution quality: {np.mean(solution_quality_list[0])}, '
          f'True solution percentage: {np.mean(solution_quality_list[1])}')

    # print(f'best_solution {best_solution}')



evol_data = []

if plot_evol_results:
    #pipeline_run(cfg, True, True, False, True)
    for model_descr in evaluation_models:
        fitting_model = check_model_config_fit(model_descr)
        if fitting_model:
            fitness_parameters = evaluation_models[model_descr]['fitness_params']
            min_approx = evaluation_models[model_descr]['min_approx']
            print(min_approx)
            loaded_ga_instance = pygad.load(evolution_file + model_descr)
            model_name = evaluation_models[model_descr]["model_name"]
            #model = model_dict[f'model{evaluation_models[model_descr]["model_name"]}']
            node_model = model_dict[f'model{model_name}'][0]
            edge_model = model_dict[f'model{model_name}'][1]

            best_solution_tuple = loaded_ga_instance.best_solution()
            # print(best_solution_tuple)
            best_solution = best_solution_tuple[0]
            model_weights_dict_node = torchga.model_weights_as_dict(model=node_model,
                                                                   weights_vector=best_solution[:node_edge_cutoff])
            node_model.load_state_dict(model_weights_dict_node)

            model_weights_dict_edge = torchga.model_weights_as_dict(model=edge_model,
                                                            weights_vector=best_solution[node_edge_cutoff:])
            edge_model.load_state_dict(model_weights_dict_edge)

            linearized_qubos, qubos, min_energy, solution_list, problem_list, \
                edge_index_list, edge_weight_list = get_training_dataset(include_loops=True)

            solution_quality_list = [[], []]
            approx_percent = []
            fitness_list = []
            for idx, (linarized_qubo, qubo, energy, solution, problem, edge_index, edge_weight) in enumerate(
                    zip(linearized_qubos, qubos, min_energy, solution_list,
                        problem_list, edge_index_list, edge_weight_list)):

                node_features = get_diagonal_of_qubo(qubo)
                node_features = node_model.forward(get_tensor_of_structure(node_features),
                                           get_tensor_of_structure(edge_index).long(),
                                           get_tensor_of_structure(edge_weight)).detach()
                approx_mask = np.zeros((qubo_size, qubo_size))
                for edge_0, edge_1 in zip(edge_index[0], edge_index[1]):
                    node_features_0 = np.array(node_features[edge_0].numpy())
                    node_features_1 = np.array(node_features[edge_1].numpy())
                    node_mean_tensor = get_tensor_of_structure(np.mean([node_features_0, node_features_1], axis=0))
                    edge_descision = edge_model.forward(node_mean_tensor).detach()
                    if edge_descision > 0:
                        approx_mask[edge_0][edge_1] = 1
                lin_approx = linearize_qubo(approx_mask)

                solution_quality, true_approx = get_quality_of_approxed_qubo(lin_approx, qubo, energy)
                approx_percent.append(true_approx / qubo_entries)
                print(solution_quality)
                solution_quality_list[0].append(solution_quality)
                solution_quality_list[1].append(np.floor(1 - solution_quality))

            evol_results = []
            for metric, results in enumerate(solution_quality_list):
                evol_results.append((np.mean(approx_percent), np.mean(results)))
            evol_data.append((evol_results, evaluation_models[model_descr]['name']))

    if evol_data:
        visualize_evol_results(aggregate_saved_problems(),
                               [i / qubo_entries for i in range(qubo_entries + 1)],
                               evol_data, solver)
