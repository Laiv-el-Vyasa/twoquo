import os

import numpy as np

from visualisation import qubo_heatmap, visualize_evol_results, plot_average_fitness
from pipeline import pipeline_run

from pygad.torchga import torchga
from torch import torch, nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn

from networks import GcnIdSimple, GcnIdStraight, GcnDiag, GcnDeep

import pygad.torchga

from evolution.evolution_util import get_training_dataset, get_fitness_value, apply_approximation_to_qubo, \
    get_quality_of_approxed_qubo, get_qubo_approx_mask, aggregate_saved_problems, solver, \
    check_model_config_fit, get_diagonal_of_qubo, cfg, get_tensor_of_structure, linearize_qubo

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

run_bool = True
restart_bool = True
extend_model = True
test_case_study = False
plot_evol_results = False


#cfg = load_cfg(cfg_id='test_evol_mc')
qubo_size = cfg['pipeline']['problems']['qubo_size']
problem = cfg['pipeline']['problems']['problems'][0]
evolution_file = f"gcn_evolution"
model_name = '_deep'
qubo_entries = int((qubo_size * (qubo_size + 1)) / 2)
population = 100
a = 1
b = 1
c = 1
d = 1
fitness_parameters = (a, b, c, d)
min_approx = 0.05


evaluation_models = {
    '_MC_8_1_05_10_1':
        {'name': 'simple model, id-matrix',
         'fitness_params': (1, .5, 10, 1),
         'min_approx': 0,
         'model_name': ''
        },
    '_MC_24_1_05_10_01':
        {'name': 'simple model, id-matrix',
         'fitness_params': (1, .5, 10, .1),
         'min_approx': 0,
         'model_name': ''
         },
    '_MC_24_1_1_1_1':
        {'name': 'simple model, id-matrix',
         'fitness_params': (1, 1, 1, 1),
         'min_approx': 0,
         'model_name': ''
         },
    '_MC_24_1_1_1_1_005':
        {'name': 'simple model, id-matrix, min .05',
         'fitness_params': (1, 1, 1, 1),
         'min_approx': .05,
         'model_name': ''
         },
    '_MC_24_straight_1_2_1_1_005':
        {'name': 'straight model, id-matrix',
         'fitness_params': (1, 2, 1, 1),
         'min_approx': .05,
         'model_name': '_straight'
         },
    '_MC_24_diag_1_1_1_1_005':
        {'name': 'single inputs, qubo-diagonal',
         'fitness_params': (1, 1, 1, 1),
         'min_approx': 0.05,
         'model_name': '_diag'
         }
}


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = geo_nn.GCNConv(qubo_size, int(qubo_size / 2), add_self_loops=False)
        self.conv2 = geo_nn.GCNConv(int(qubo_size / 2), qubo_size, add_self_loops=False)

    def forward(self, x, edge_index, edge_weights):
        x = self.conv1(x, edge_index, edge_weights)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weights)
        return F.relu(x)


model = Network()

model_dict = {
    'model': model,
    'model_straight': GcnIdStraight(qubo_size),
    'model_diag': GcnDiag(qubo_size, 5),
    'model_deep': GcnDeep(qubo_size)
}

pytorch_total_params = sum(p.numel() for p in model_dict[f'model{model_name}'].parameters() if p.requires_grad)
#print(model_name, pytorch_total_params)

torch_ga = pygad.torchga.TorchGA(model=model_dict[f'model{model_name}'], num_solutions=population)

avg_fitness_list = []
avg_fitness_generation = []


def fitness_func(solution, solution_idx):
    global torch_ga, model, avg_fitness_list, avg_fitness_generation, model_dict, model_name, min_approx

    model = model_dict[f'model{model_name}']
    #print(model)

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)

    # Use the current solution as the model parameters.
    model.load_state_dict(model_weights_dict)

    include_loops = not (model_name == '_diag' or model_name == '_deep')
    _, qubos, min_energy, _, _, edge_index_list, edge_weight_list = get_training_dataset(include_loops=include_loops)
    #linearized_approx = model(linearized_qubos).detach()
    linearized_approx = []
    for qubo, edge_index, edge_weight in zip(qubos, edge_index_list, edge_weight_list):
        #print('QUBO:', qubo)
        if model_name == '_diag':
            node_features = get_diagonal_of_qubo(qubo)
        elif model_name == '_deep':
            node_features = qubo
            edge_weight = []
        else:
            node_features = np.identity(qubo_size)
        approx_mask = model.forward(get_tensor_of_structure(node_features),
                                    get_tensor_of_structure(edge_index).long(),
                                    get_tensor_of_structure(edge_weight))
        #print('APPROX', approx_mask)
        linearized_approx.append(linearize_qubo(approx_mask.detach()))

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
initial_population = torch_ga.population_weights

arr1 = [[1, 2], [1, 4]]
arr2 = [[3, 4], [1, 5]]

print('Extension: ', np.append(arr1, arr2, axis=1))

arr3 = [1.5, 3.2]
arr4 = [0.5, -1.8]
print([arr3, arr4])
print(np.mean([arr3, arr4], axis=0))

#print(initial_population)
#print(len(initial_population[0]))

ga_instance = pygad.GA(num_generations=num_generations,
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


test_model = '_MC_24_diag_1_1_1_1_005'
test_cases = 5
if test_case_study:
    fitness_parameters = evaluation_models[test_model]['fitness_params']
    min_approx = evaluation_models[test_model]['min_approx']
    model_name = evaluation_models[test_model]["model_name"]
    model = model_dict[f'model{evaluation_models[test_model]["model_name"]}']
    loaded_ga_instance = pygad.load(evolution_file + test_model)
    #loaded_ga_instance = pygad.load(evolution_file)

    best_solution_tuple = loaded_ga_instance.best_solution()
    print('Best solution tuple: ', best_solution_tuple)
    best_solution = best_solution_tuple[0]
    best_model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                            weights_vector=best_solution)
    print('Weight dict: ', best_model_weights_dict)
    model.load_state_dict(best_model_weights_dict)
    include_loops = not (model_name == '_diag' or model_name == '_deep')
    linearized_qubos, qubos, min_energy, solution_list, problem_list, \
        edge_index_list, edge_weight_list = get_training_dataset(include_loops=include_loops)
    #linearized_approx = model(linearized_qubos).detach()

    solution_quality_list = [[], []]
    approx_quality_list = []
    fitness_list = []
    for idx, (linarized_qubo, qubo, energy, solution, problem, edge_index, edge_weight) in enumerate(
            zip(linearized_qubos, qubos, min_energy, solution_list,
                problem_list, edge_index_list, edge_weight_list)):

        if model_name == '_diag':
            node_features = get_diagonal_of_qubo(qubo)
        elif model_name == '_deep':
            node_features = qubo
            edge_weight = []
        else:
            node_features = np.identity(qubo_size)

        approx_mask = model.forward(get_tensor_of_structure(node_features),
                                    get_tensor_of_structure(edge_index).long(),
                                    get_tensor_of_structure(edge_weight))
        lin_approx = linearize_qubo(approx_mask.detach())

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
            model = model_dict[f'model{evaluation_models[model_descr]["model_name"]}']

            best_solution_tuple = loaded_ga_instance.best_solution()
            # print(best_solution_tuple)
            best_solution = best_solution_tuple[0]
            best_model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                                    weights_vector=best_solution)
            model.load_state_dict(best_model_weights_dict)

            #linearized_qubos, qubos, min_energy, solution_list, problem_list = get_training_dataset()
            #linearized_approx = model(linearized_qubos).detach()

            solution_quality_list = [[], []]
            approx_percent = []

            include_loops = not (model_name == '_diag' or model_name == '_deep')
            linearized_qubos, qubos, min_energy, solution_list, problem_list, \
                edge_index_list, edge_weight_list = get_training_dataset(include_loops=include_loops)
            #linearized_approx = model(linearized_qubos).detach()

            fitness_list = []
            for idx, (linarized_qubo, qubo, energy, solution, problem, edge_index, edge_weight) in enumerate(
                    zip(linearized_qubos, qubos, min_energy, solution_list,
                        problem_list, edge_index_list, edge_weight_list)):

                if model_name == '_diag':
                    node_features = get_diagonal_of_qubo(qubo)
                elif model_name == '_deep':
                    node_features = qubo
                    edge_weight = []
                else:
                    node_features = np.identity(qubo_size)

                approx_mask = model.forward(get_tensor_of_structure(node_features),
                                            get_tensor_of_structure(edge_index).long(),
                                            get_tensor_of_structure(edge_weight))
                lin_approx = linearize_qubo(approx_mask.detach())

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
