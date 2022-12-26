import os

import numpy as np

from visualisation import qubo_heatmap, visualize_evol_results, plot_average_fitness
from pipeline import pipeline_run

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from pygad.torchga import torchga
from torch import torch, nn
print('Torch: ', torch.__version__)
print('Cuda: ', torch.version.cuda)

from config import load_cfg
import pygad.torchga

from evolution.evolution_util import get_training_dataset, get_fitness_value, apply_approximation_to_qubo, \
    get_quality_of_approxed_qubo, get_qubo_approx_mask, aggregate_saved_problems, solver, \
    check_model_config_fit, cfg

run_bool = False
restart_bool = False
test_case_study = False
plot_evol_results = False


#cfg = load_cfg(cfg_id='test_evol_mc')
qubo_size = cfg['pipeline']['problems']['qubo_size']
problem = cfg['pipeline']['problems']['problems'][0]
evolution_file = f"simple_evolution"
model_name = '_autoenc'
qubo_entries = int((qubo_size * (qubo_size + 1)) / 2)
population = 100
a = 1
b = .5
c = 10
d = .1
fitness_parameters = (a, b, c, d)


evaluation_models = {
    '_NP_8_1_1_0_05':
        {'name': 'quality, target .5 approxed entries',
         'fitness_params': (1, 1, 0, .5),
         'model_name': ''
         },
    '_NP_8_1_1_0_08':
        {'name': 'quality, target .8 approxed entries',
         'fitness_params': (1, 1, 0, .8),
         'model_name': ''
         },
    '_NP_8_1_05_10_1':
        {'name': 'correct_solutions',
         'fitness_params': (1, .5, 10, 1),
         'model_name': ''
         },
    '_NP_24_1_05_10_01_20p': {
        'name': 'correct solutions, 20 parents',
        'fitness_params': (1, .5, 10, .1),
        'model_name': ''
    },
    '_NP_24_1_1_0_09_20p': {
        'name': 'quality, target .9 approxed entries',
        'fitness_params': (1, 1, 0, .9),
        'model_name': ''
    },
    '_MC_8_1_05_10_1_MC': {
        'name': 'max-cut, correct solutions',
        'fitness_params': (1, .5, 10, 1),
        'model_name': ''
    },
    '_MC_24_1_05_10_1': {
        'name': 'simple model (max approx)',
        'fitness_params': (1, .5, 10, 1),
        'model_name': ''
    },
    '_MC_24_1_05_10_01': {
        'name': 'simple model',
        'fitness_params': (1, .5, 10, .1),
        'model_name': ''
    },
    '_MC_24_autoenc_1_05_10_01': {
        'name': 'autoencoder model',
        'fitness_params': (1, .5, 10, .1),
        'model_name': '_autoenc'
    },
    '_MC_24_sqrt_autoenc_1_05_10_01': {
        'name': 'sqrt autoencoder model',
        'fitness_params': (1, .5, 10, .1),
        'model_name': '_sqrt_autoenc'
    }
}


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(qubo_entries, qubo_entries)
        self.linear2 = nn.Linear(qubo_entries, qubo_entries)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return nn.ReLU(x)


# model = Network()
model = nn.Sequential(
    nn.Linear(qubo_entries, qubo_entries),
    nn.Linear(qubo_entries, qubo_entries),
    nn.ReLU()
)

model_autoenc = nn.Sequential(
    nn.Linear(qubo_entries, int(qubo_entries / 2)),
    nn.Linear(int(qubo_entries/2), int(qubo_entries/4)),
    nn.Sigmoid(),
    nn.Linear(int(qubo_entries/4), int(qubo_entries/2)),
    nn.Linear(int(qubo_entries/2), qubo_entries),
    nn.ReLU()
)

model_sqrt_autoenc = nn.Sequential(
    nn.Linear(qubo_entries, int(np.sqrt(qubo_entries))),
    nn.Linear(int(np.sqrt(qubo_entries)), int(np.sqrt(qubo_entries) / 2)),
    nn.Sigmoid(),
    nn.Linear(int(np.sqrt(qubo_entries) / 2), int(np.sqrt(qubo_entries))),
    nn.Linear(int(np.sqrt(qubo_entries)), qubo_entries),
    nn.ReLU()
)

model_dict = {
    'model': model,
    'model_autoenc': model_autoenc,
    'model_sqrt_autoenc': model_sqrt_autoenc
}

for mdl_name in model_dict:
    mdl = model_dict[mdl_name]
    pytorch_total_params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(mdl_name, pytorch_total_params)

torch_ga = pygad.torchga.TorchGA(model=model_dict[f'model{model_name}'], num_solutions=population)


avg_fitness_list = []
avg_fitness_generation = []


def fitness_func(solution, solution_idx):
    global data_inputs, data_outputs, torch_ga, model, avg_fitness_list, avg_fitness_generation, model_dict, model_name
    model = model_dict[f'model{model_name}']

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)

    # Use the current solution as the model parameters.
    model.load_state_dict(model_weights_dict)

    linearized_qubos, qubos, min_energy, *_ = get_training_dataset()
    linearized_approx = model(linearized_qubos).detach()

    solution_fitness = get_fitness_value(linearized_approx, qubos, min_energy, fitness_parameters)
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


num_generations = 200
num_parents_mating = int(population * .2)
initial_population = torch_ga.population_weights

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


test_model = '_MC_24_1_05_10_1'
test_cases = 5
if test_case_study:
    fitness_parameters = evaluation_models[test_model]['fitness_params']
    model_name = evaluation_models[test_model]["model_name"]
    model = model_dict[f'model{evaluation_models[test_model]["model_name"]}']
    loaded_ga_instance = pygad.load(evolution_file + test_model)
    #loaded_ga_instance = pygad.load(evolution_file)

    best_solution_tuple = loaded_ga_instance.best_solution()
    print(best_solution_tuple)
    best_solution = best_solution_tuple[0]
    best_model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                            weights_vector=best_solution)
    model.load_state_dict(best_model_weights_dict)

    linearized_qubos, qubos, min_energy,\
        solution_list, problem_list, edge_index_list, edge_weight_list = get_training_dataset()
    linearized_approx = model(linearized_qubos).detach()

    solution_quality_list = [[], []]
    approx_quality_list = []
    fitness_list = []
    for idx, (lin_approx, qubo, energy, solution, problem) in enumerate(
            zip(linearized_approx, qubos, min_energy,
                solution_list, problem_list)):
        approxed_qubo, true_approx = apply_approximation_to_qubo(lin_approx, qubo)
        solution_quality, _ = get_quality_of_approxed_qubo(lin_approx, qubo, energy, print_solutions=True)
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
            loaded_ga_instance = pygad.load(evolution_file + model_descr)
            model_name = evaluation_models[model_descr]["model_name"]
            model = model_dict[f'model{evaluation_models[model_descr]["model_name"]}']

            best_solution_tuple = loaded_ga_instance.best_solution()
            # print(best_solution_tuple)
            best_solution = best_solution_tuple[0]
            best_model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                                weights_vector=best_solution)
            model.load_state_dict(best_model_weights_dict)

            linearized_qubos, qubos, min_energy, solution_list, problem_list,\
                edge_index_list, edge_weight_list = get_training_dataset()
            linearized_approx = model(linearized_qubos).detach()

            solution_quality_list = [[], []]
            approx_percent = []

            for lin_approx, qubo, energy in zip(linearized_approx, qubos, min_energy):
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
