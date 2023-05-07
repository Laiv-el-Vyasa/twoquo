import numpy as np

from approximation import get_approximated_qubos
from config import load_cfg
from evolution_new.combined_evolution_training import get_data_from_training_config
from evolution_new.evolution_utils import get_quality_of_approxed_qubo, get_qubo_approx_mask, get_file_name, \
    get_relative_size_of_approxed_entries, get_classical_solution_qualities, get_min_solution_quality
from evolution_new.pygad_learning import PygadLearner
from evolution_new.learning_model import LearningModel
from new_visualisation import visualisation_pipeline, qubo_heatmap, visualize_boxplot_comparison

solver = 'qbsolv_simulated_annealing'


class TrainingAnalysis:
    def __init__(self, config_name: str, analysis_parameters: dict):
        self.model, learning_parameters, fitness_func = get_data_from_training_config(config_name)
        self.pygad_learner = PygadLearner(self.model, learning_parameters, fitness_func)
        self.config = load_cfg(cfg_id=learning_parameters['config_name'])
        self.config["solvers"][solver]['repeats'] = 100
        print(self.config["solvers"]['qbsolv_simulated_annealing']['repeats'])
        self.analysis_name = get_file_name(analysis_parameters['analysis_name'], self.config,
                                           learning_parameters['fitness_parameters'], analysis=True)
        self.config['pipeline']['problems']['n_problems'] *= 1
        self.analysis_parameters = analysis_parameters
        if not self.model.load_best_model(learning_parameters['training_name']):
            self.pygad_learner.save_best_model()
            self.model.load_best_model(learning_parameters['training_name'])

    def run_analysis(self):
        approximation_quality_dict = self.get_model_approximation_quality()
        analysis_baseline = self.get_analysis_baseline()
        self.create_visualisation_calls(analysis_baseline, approximation_quality_dict)

    def get_model_approximation_quality(self) -> dict:
        return_dict = {
            'solution_quality_list': [],
            'min_solution_quality_list': [],
            'mean_solution_quality_list': [],
            'min_mean_solution_quality_list': [],
            'mean_mean_solution_quality_list': [],
            'approx_percent_list': [],
            'correct_approx_list': [],
            'incorrect_approx_list': [],
            'correct_approx_size': [],
            'incorrect_approx_size': [],
            'classical_min_solution_quality': [],
            'classical_mean_solution_quality': [],
            'repeat_qubo_min_solution_quality': [],
            'repeat_qubo_mean_solution_quality': []
        }
        problem_dict = self.model.get_approximation(self.model.get_training_dataset(self.config))
        approx_qubo_list, solutions_list, qubo_list, problem_list = problem_dict['approxed_qubo_list'], \
                                                                    problem_dict['solutions_list'], \
                                                                    problem_dict['qubo_list'], \
                                                                    problem_dict['problem_list']

        for idx, (qubo, approx_qubo, solutions, problem) in enumerate(zip(qubo_list, approx_qubo_list,
                                                                          solutions_list, problem_list)):
            print(f'Approximating problem {idx} via model')
            # print(solutions)
            if idx < self.analysis_parameters['show_qubo_mask']:
                qubo_heatmap(qubo)
                qubo_heatmap(get_qubo_approx_mask(approx_qubo, qubo))
            min_solution_quality, _, approx_percent, mean_solution_quality, min_mean_sol_qual, mean_mean_sol_qual \
                = get_quality_of_approxed_qubo(qubo, approx_qubo, solutions, self.config)
            return_dict['solution_quality_list'].append((np.floor(1 - min_solution_quality)))
            return_dict['min_solution_quality_list'].append((1 - min_solution_quality))
            return_dict['mean_solution_quality_list'].append((1 - mean_solution_quality))
            return_dict['min_mean_solution_quality_list'].append((1 - min_mean_sol_qual))
            return_dict['mean_mean_solution_quality_list'].append((1 - min_solution_quality))
            approx_size = get_relative_size_of_approxed_entries(approx_qubo, qubo)
            # print('approx size: ', approx_size)
            # print(min_solution_quality, approx_percent)
            if min_solution_quality <= 0 and approx_percent != 0:
                # print('True solution found')
                return_dict['correct_approx_list'].append(approx_percent)
                return_dict['correct_approx_size'].append(approx_size)
            else:
                return_dict['incorrect_approx_list'].append(approx_percent)
                return_dict['incorrect_approx_size'].append(approx_size)
            return_dict['approx_percent_list'].append(approx_percent)

            if 'compare_different_approaches' in self.analysis_parameters and \
                    self.analysis_parameters['compare_different_approaches']:
                classical_min_solution_quality, classical_mean_solution_quality = \
                    get_classical_solution_qualities(solutions, qubo, problem,
                                                     self.config["solvers"]['qbsolv_simulated_annealing']['repeats'])
                return_dict['classical_min_solution_quality'].append(classical_min_solution_quality)
                return_dict['classical_mean_solution_quality'].append(classical_mean_solution_quality)

                repeat_qubo_min_solution_quality, _, repeat_qubo_mean_solution_quality, *_ = \
                    get_min_solution_quality(solutions, qubo, solutions)
                return_dict['repeat_qubo_min_solution_quality'].append(repeat_qubo_min_solution_quality)
                return_dict['repeat_qubo_mean_solution_quality'].append(repeat_qubo_mean_solution_quality)
        return return_dict

    def get_analysis_baseline(self) -> list[list, list, list, list, list, list, list]:
        try:
            analysis_baseline = np.load(f'analysis_baseline/{self.analysis_name}.npy')
            print('Analysis baseline loaded')
        except FileNotFoundError:
            analysis_baseline = self.get_new_analysis_baseline()
            np.save(f'analysis_baseline/{self.analysis_name}', analysis_baseline)
        print(analysis_baseline)
        return analysis_baseline

    def get_new_analysis_baseline(self) -> list[list, list, list, list, list, list, list]:
        analysis_baseline = [[], [], [], [], [], [], []]
        problem_dict = self.model.get_training_dataset(self.config)
        stepwise_approx_quality, stepwise_min_approx_quality, stepwise_mean_approx_quality, \
            stepwise_approx_quality_random, stepwise_min_approx_quality_random, stepwise_mean_approx_quality_random \
            = self.get_stepwise_approx_quality(problem_dict)
        # Prepare array for saving and display
        analysis_baseline[0] = stepwise_approx_quality
        step_list = [n / (self.analysis_parameters['steps'] + 1) for n in range(self.analysis_parameters['steps'] + 1)]
        analysis_baseline[1] = step_list
        analysis_baseline[3] = stepwise_min_approx_quality
        analysis_baseline[4] = stepwise_mean_approx_quality
        analysis_baseline[2] = stepwise_approx_quality_random
        analysis_baseline[5] = stepwise_min_approx_quality_random
        analysis_baseline[6] = stepwise_mean_approx_quality_random
        return analysis_baseline

    def get_stepwise_approx_quality(self, problem_dict: dict) -> tuple[list, list, list, list, list, list]:
        qubo_list, solutions_list = problem_dict['qubo_list'], problem_dict['solutions_list']
        solution_quality_list = []
        min_solution_quality_list = []
        mean_solution_quality_list = []
        random_solution_quality_list = []
        random_min_solution_quality_list = []
        random_mean_solution_quality_list = []
        for idx, (qubo, solutions) in enumerate(zip(qubo_list, solutions_list)):
            print(f'Approximating problem {idx} for baseline')
            sol_qual_sorted, min_sol_qual_sorted, mean_sol_qual_sorted = \
                self.get_stepwise_approx_quality_for_qubo(qubo, solutions, True)
            #print(sol_qual_sorted)
            solution_quality_list.append(sol_qual_sorted)
            min_solution_quality_list.append(min_sol_qual_sorted)
            mean_solution_quality_list.append(mean_sol_qual_sorted)
            sol_qual_random, min_sol_qual_random, mean_sol_qual_random = \
                self.get_stepwise_approx_quality_for_qubo(qubo, solutions, False)
            random_solution_quality_list.append(sol_qual_random)
            random_min_solution_quality_list.append(min_sol_qual_random)
            random_mean_solution_quality_list.append(mean_sol_qual_random)
        print('sol_qual_list ', solution_quality_list)
        return self.rotate_solution_quality_list(solution_quality_list), \
               self.rotate_solution_quality_list(min_solution_quality_list), \
               self.rotate_solution_quality_list(mean_solution_quality_list), \
               self.rotate_solution_quality_list(random_solution_quality_list), \
               self.rotate_solution_quality_list(random_min_solution_quality_list), \
               self.rotate_solution_quality_list(random_mean_solution_quality_list)

    def get_stepwise_approx_quality_for_qubo(self, qubo: list, solutions: list, sorted_approx: bool) \
            -> tuple[list, list, list]:
        stepwise_approx_quality = [1.]
        stepwise_min_approx_quality = [1.]
        stepwise_mean_approx_quality = [1.]
        approximation_dict, _ = get_approximated_qubos(qubo, False, True, self.analysis_parameters['steps'],
                                                       sorted_approx=sorted_approx)
        for i in range(self.analysis_parameters['steps']):
            approx_qubo = approximation_dict[str(i + 1)]['qubo']
            min_solution_quality, _, _, mean_solution_quality, *_ \
                = get_quality_of_approxed_qubo(qubo, approx_qubo, solutions, self.config)
            #print('step ', i)
            #print(min_solution_quality)
            stepwise_approx_quality.append(np.floor(1 - min_solution_quality))
            stepwise_min_approx_quality.append((1 - min_solution_quality))
            stepwise_mean_approx_quality.append((1 - mean_solution_quality))
        return stepwise_approx_quality, stepwise_min_approx_quality, stepwise_mean_approx_quality

    def rotate_solution_quality_list(self, solution_quality_list: list[list]) -> list:
        rotated_list = [[] for n in range(self.analysis_parameters['steps'] + 1)]
        analysis_baseline = []
        for problem_data in solution_quality_list:
            for id_x, approx_step_quality in enumerate(problem_data):
                rotated_list[id_x].append(approx_step_quality)
        for approx_step_quality_list in rotated_list:
            analysis_baseline.append(np.mean(approx_step_quality_list))
        return analysis_baseline

    def get_visualisation_title(self, evaluation_type, models=False):
        problems = self.config["pipeline"]["problems"]["problems"]
        title = f'{evaluation_type}, trained model{"s" if models else ""}, problem{"s" if len(problems) > 1 else ""}: '
        for problem in problems:
            title = title + problem + ', '
        title = title + f'Max-size: {self.config["pipeline"]["problems"]["qubo_size"]}, Solver: ' \
                        f'{solver}'
        return title

    def create_baseline_data_dict(self, baseline_data: list[list, list, list, list, list, list, list],
                                  sorted_approx=True, data_index=0, dotted=False, color='black') -> dict:
        return {
            'percent_list': baseline_data[1],
            'color': color,
            'dotted': dotted,
            'baseline_approx_data': baseline_data[data_index],
            'legend': f'stepwise-approximation: {self.analysis_parameters["steps"]} steps, '
                      f'{"smallest entries first" if sorted_approx else "random entries"}, '
                      f'{"mean" if data_index == 3 or data_index == 6 else "best"}'
        }

    def create_visualisation_calls(self, analysis_baseline: list[list, list, list, list, list, list, list],
                                   approximation_quality_dict: dict, size_analysis=False):
        visualisation_pipeline({
            'baseline_data': [self.create_baseline_data_dict(analysis_baseline)],
            'evaluation_results': [
                {
                    'color': 'black',
                    'marker': 4,
                    'evol_y': [np.mean(approximation_quality_dict['solution_quality_list'])],
                    'evol_x': [np.mean(approximation_quality_dict['approx_percent_list'])],
                    'label': 'Avg. solution quality, avg. approx percent of model'
                }
            ],
            'title': self.get_visualisation_title('Solution quality'),
            'x_label': 'approximated qubo entries in percent',
            'y_label': 'solution quality'
        })
        visualisation_pipeline({
            'baseline_data': [self.create_baseline_data_dict(analysis_baseline)],
            'evaluation_results': [
                {
                    'color': 'green',
                    'marker': 4,
                    'evol_y': [1 for _ in approximation_quality_dict['correct_approx_list']],
                    'evol_x': approximation_quality_dict['correct_approx_list'],
                    'label': 'Correct solutions suggested by model'
                },
                {
                    'color': 'black',
                    'marker': 4,
                    'evol_y': [0 for _ in approximation_quality_dict['incorrect_approx_list']],
                    'evol_x': approximation_quality_dict['incorrect_approx_list'],
                    'label': 'Incorrect solutions suggested by model'
                },
            ],
            'title': self.get_visualisation_title('Solution quality of every solution'),
            'x_label': 'approximated qubo entries in percent',
            'y_label': 'solution quality'
        })
        visualisation_pipeline({  # Sorted/random with min/mean pipeline
            'baseline_data': [
                self.create_baseline_data_dict(analysis_baseline, data_index=3),
                self.create_baseline_data_dict(analysis_baseline, data_index=4, dotted=True),
                self.create_baseline_data_dict(analysis_baseline, data_index=5, sorted_approx=False,
                                               color='grey'),
                self.create_baseline_data_dict(analysis_baseline, data_index=6, sorted_approx=False,
                                               dotted=True, color='grey')
            ],
            'evaluation_results': [
                {
                    'color': 'blue',
                    'marker': 4,
                    'alpha': 0.7,
                    'evol_y': approximation_quality_dict['min_solution_quality_list'],
                    'evol_x': approximation_quality_dict['approx_percent_list'],
                    'label': 'Min solution quality of approximation suggested by model'
                },
                {
                    'color': 'slateblue',
                    'marker': 5,
                    'alpha': 0.7,
                    'evol_y': approximation_quality_dict['mean_solution_quality_list'],
                    'evol_x': approximation_quality_dict['approx_percent_list'],
                    'label': 'Mean solution quality of approximation suggested by model'
                },
            ],
            'title': self.get_visualisation_title('Solution quality of every solution'),
            'x_label': 'approximated qubo entries in percent',
            'y_label': 'solution quality (min energy - energy) / min energy'
        })
        if self.analysis_parameters['size_analysis']:
            visualisation_pipeline({
                'evaluation_results': [
                    {
                        'color': 'blue',
                        'marker': 4,
                        'evol_y': approximation_quality_dict['correct_approx_size'],
                        'evol_x': approximation_quality_dict['correct_approx_list'],
                        'label': 'Size of approximated entries of correct solutions'
                    },
                    {
                        'color': 'red',
                        'marker': 4,
                        'evol_y': approximation_quality_dict['incorrect_approx_size'],
                        'evol_x': approximation_quality_dict['incorrect_approx_list'],
                        'label': 'Size of approximated entries of incorrect solutions'
                    },
                ],
                'title': self.get_visualisation_title('Relative size of approximated entries'),
                'x_label': 'approximated qubo entries in percent',
                'y_label': 'Cumulated size of approximated entries (1: n biggest, 0: n smallest)',
                'scale_axis': True
            })
        if 'compare_different_approaches' in self.analysis_parameters and \
                self.analysis_parameters['compare_different_approaches']:
            visualize_boxplot_comparison({
                'data_list':
                    [
                        {
                            'min': approximation_quality_dict['min_solution_quality_list'],
                            'mean': approximation_quality_dict['mean_solution_quality_list'],
                            'tick_name': 'Model performance'
                        },
                        {
                            'min': approximation_quality_dict['classical_min_solution_quality'],
                            'mean': approximation_quality_dict['classical_mean_solution_quality'],
                            'tick_name': 'Classical algorithm'
                        },
                        {
                            'min': approximation_quality_dict['repeat_qubo_min_solution_quality'],
                            'mean': approximation_quality_dict['repreat_qubo_mean_solution_quality'],
                            'tick_name': 'Original QUBO'
                        }
                    ],
                'colors':
                    {
                        'min': '#D7191C',
                        'mean': '#2C7BB6'
                    },
                'y_label': 'solution quality (min energy - energy) / min energy',
                'title': self.get_visualisation_title('Comparison of different approaches')
            })
