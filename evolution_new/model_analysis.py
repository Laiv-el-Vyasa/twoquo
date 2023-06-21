import numpy as np

from approximation import get_approximated_qubos
from config import load_cfg
from evolution_new.combined_evolution_training import get_data_from_training_config
from evolution_new.evolution_utils import get_quality_of_approxed_qubo, get_qubo_approx_mask, get_file_name, \
    get_relative_size_of_approxed_entries, get_classical_solution_qualities, get_min_solution_quality, \
    remove_hard_constraits_from_qubo, delete_data, get_approximation_count, get_analysis_results_file_name
from evolution_new.pygad_learning import PygadLearner
from new_visualisation import qubo_heatmap
from combined_model_features_onehot import CombinedOneHotFeatureModel


class ModelAnalysis:
    def __init__(self, config_name: str, config_name_list: list, analysis_parameters: dict):
        self.solver = analysis_parameters['solver']
        self.model_result_list = []
        self.model, learning_parameters, fitness_func = get_data_from_training_config(config_name)
        self.pygad_learner = PygadLearner(self.model, learning_parameters, fitness_func)
        self.analysis_parameters = analysis_parameters
        self.learning_parameters = learning_parameters
        self.config_list = self.get_config_list(config_name_list)
        #self.config = load_cfg(cfg_id=learning_parameters['config_name'])
        #self.config["solvers"][solver]['repeats'] = 100
        #self.config["solvers"][solver]['enabled'] = True
        #print(self.config["solvers"]['qbsolv_simulated_annealing']['repeats'])
        #self.analysis_name = get_file_name(analysis_parameters['analysis_name'], self.config,
        #                                   learning_parameters['fitness_parameters'], analysis=True)
        #self.config['pipeline']['problems']['n_problems'] *= 1
        if not self.model.load_best_model(learning_parameters['training_name']):
            self.pygad_learner.save_best_model()
            self.model.load_best_model(learning_parameters['training_name'])

    def get_config_list(self, config_name_list: list) -> list:
        config_list = []
        for idx, config_name in enumerate(config_name_list):
            if config_name == 'standard':
                config = load_cfg(cfg_id=self.learning_parameters['config_name'])
            else:
                config = load_cfg(cfg_id=config_name)
            config["solvers"][self.solver]['repeats'] = 100
            config["solvers"][self.solver]['enabled'] = True
            config['pipeline']['problems']['n_problems'] *= 1
            if 'scale_list' in self.analysis_parameters:
                config['pipeline']['problems']['scale']['min'] = self.analysis_parameters['scale_list'][idx]
                config['pipeline']['problems']['scale']['max'] = self.analysis_parameters['scale_list'][idx]
            config_list.append(config)
        return config_list

    def run_analysis(self):
        for config in self.config_list:
            delete_data()
            approximation_quality_dict = self.get_model_approximation_quality(config)
            analysis_baseline = self.get_analysis_baseline(config)
            self.model_result_list.append({
                'approximation_quality_dict': approximation_quality_dict,
                'baseline': analysis_baseline
            })

    def get_model_approximation_dict(self, config: dict) -> dict:
        filename = get_analysis_results_file_name(self.analysis_parameters['analysis_name'],
                                                  load_cfg(cfg_id=self.learning_parameters['config_name']),
                                                  config,
                                                  self.learning_parameters['fitness_parameters'])
        try:
            approximation_dict = np.load(f'analysis_results/{filename}.npy', allow_pickle=True).item()
            print('Analysis results loaded')
        except FileNotFoundError:
            approximation_dict = self.get_model_approximation_dict(config)
            np.save(f'analysis_results/{filename}', approximation_dict)
        # print(approximation_dict)
        return approximation_dict

    def get_model_approximation_quality(self, config: dict) -> dict:
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
            'classical_solution_quality': [],
            'classical_min_solution_quality': [],
            'classical_mean_solution_quality': [],
            'random_solution_quality': [],
            'random_min_solution_quality': [],
            'random_mean_solution_quality': [],
            'repeat_qubo_min_solution_quality': [],
            'repeat_qubo_mean_solution_quality': []
        }
        problem_dict = self.model.get_approximation(self.model.get_training_dataset(config))
        approx_qubo_list, solutions_list, qubo_list, problem_list = problem_dict['approxed_qubo_list'], \
                                                                    problem_dict['solutions_list'], \
                                                                    problem_dict['qubo_list'], \
                                                                    problem_dict['problem_list']

        for idx, (qubo, approx_qubo, solutions, problem) in enumerate(zip(qubo_list, approx_qubo_list,
                                                                          solutions_list, problem_list)):
            print(f'Approximating problem {idx} via model')
            if idx < self.analysis_parameters['show_qubo_mask']:
                qubo_heatmap(qubo)
                qubo_heatmap(get_qubo_approx_mask(approx_qubo, qubo))
            min_solution_quality, _, approx_percent, mean_solution_quality, min_mean_sol_qual, mean_mean_sol_qual \
                = get_quality_of_approxed_qubo(qubo, approx_qubo, solutions, config)
            if isinstance(self.model.__class__, CombinedOneHotFeatureModel.__class__):
                qubo_to_approx = remove_hard_constraits_from_qubo(qubo, problem, True)
                _, percent_hard_contraints = get_approximation_count(qubo, qubo_to_approx)
                if percent_hard_contraints == 1:
                    approx_percent = 0
                else:
                    approx_percent = approx_percent / (1 - percent_hard_contraints)

            return_dict['solution_quality_list'].append((np.floor(1 - min_solution_quality)))
            return_dict['min_solution_quality_list'].append(min_solution_quality)
            return_dict['mean_solution_quality_list'].append(mean_solution_quality)
            return_dict['min_mean_solution_quality_list'].append(min_mean_sol_qual)
            return_dict['mean_mean_solution_quality_list'].append(min_solution_quality)
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
                repeats = config["solvers"][self.solver]['repeats']
                classical_min_solution_quality, classical_mean_solution_quality = \
                    get_classical_solution_qualities(solutions, qubo, problem, repeats, False)
                return_dict['classical_solution_quality'].append(np.floor(1 - classical_min_solution_quality))
                return_dict['classical_min_solution_quality'].append(classical_min_solution_quality)
                return_dict['classical_mean_solution_quality'].append(classical_mean_solution_quality)

                random_min_solution_quality, random_mean_solution_quality = \
                    get_classical_solution_qualities(solutions, qubo, problem, repeats, True)
                return_dict['random_solution_quality'].append(np.floor(1 - random_min_solution_quality))
                return_dict['random_min_solution_quality'].append(random_min_solution_quality)
                return_dict['random_mean_solution_quality'].append(random_mean_solution_quality)

                repeat_qubo_min_solution_quality, _, repeat_qubo_mean_solution_quality, *_ = \
                    get_min_solution_quality(solutions, qubo, solutions)
                # print(repeat_qubo_min_solution_quality, repeat_qubo_mean_solution_quality)
                return_dict['repeat_qubo_min_solution_quality'].append(repeat_qubo_min_solution_quality)
                return_dict['repeat_qubo_mean_solution_quality'].append(repeat_qubo_mean_solution_quality)
        return return_dict

    def get_analysis_baseline(self, config) -> list[list, list, list, list, list, list, list]:
        analysis_name = get_file_name(self.analysis_parameters['analysis_name'], config,
                                      self.learning_parameters['fitness_parameters'], analysis=True,
                                      steps=self.analysis_parameters['steps'])
        try:
            analysis_baseline = np.load(f'analysis_baseline/{analysis_name}.npy')
            print('Analysis baseline loaded')
        except FileNotFoundError:
            analysis_baseline = self.get_new_analysis_baseline(config)
            np.save(f'analysis_baseline/{analysis_name}', analysis_baseline)
        print(analysis_baseline)
        return analysis_baseline

    def get_new_analysis_baseline(self, config: dict) -> list[list, list, list, list, list, list, list]:
        analysis_baseline = [[], [], [], [], [], [], []]
        problem_dict = self.model.get_training_dataset(config)
        stepwise_approx_quality, stepwise_min_approx_quality, stepwise_mean_approx_quality, \
        stepwise_approx_quality_random, stepwise_min_approx_quality_random, stepwise_mean_approx_quality_random \
            = self.get_stepwise_approx_quality(problem_dict, config)
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

    def get_stepwise_approx_quality(self, problem_dict: dict, config: dict) \
            -> tuple[list, list, list, list, list, list]:
        qubo_list, solutions_list, problem_list = problem_dict['qubo_list'], problem_dict['solutions_list'], \
                                                  problem_dict['problem_list']
        solution_quality_list = []
        min_solution_quality_list = []
        mean_solution_quality_list = []
        random_solution_quality_list = []
        random_min_solution_quality_list = []
        random_mean_solution_quality_list = []
        for idx, (qubo, solutions, problem) in enumerate(zip(qubo_list, solutions_list, problem_list)):
            print(f'Approximating problem {idx} for baseline')
            sol_qual_sorted, min_sol_qual_sorted, mean_sol_qual_sorted = \
                self.get_stepwise_approx_quality_for_qubo(qubo, solutions, config, problem, True)
            # print(sol_qual_sorted)
            solution_quality_list.append(sol_qual_sorted)
            min_solution_quality_list.append(min_sol_qual_sorted)
            mean_solution_quality_list.append(mean_sol_qual_sorted)
            sol_qual_random, min_sol_qual_random, mean_sol_qual_random = \
                self.get_stepwise_approx_quality_for_qubo(qubo, solutions, config, problem, False)
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

    def get_stepwise_approx_quality_for_qubo(self, qubo: np.array, solutions: list, config: dict, problem: dict,
                                             sorted_approx: bool) -> tuple[list, list, list]:
        stepwise_approx_quality = [1.]
        stepwise_min_approx_quality = [1.]
        stepwise_mean_approx_quality = [1.]
        if isinstance(self.model.__class__, CombinedOneHotFeatureModel.__class__):
            qubo_to_approx = remove_hard_constraits_from_qubo(qubo, problem, True)
        else:
            qubo_to_approx = qubo
        approximation_dict, _ = get_approximated_qubos(qubo_to_approx, False, True, self.analysis_parameters['steps'],
                                                       sorted_approx=sorted_approx)
        for i in range(self.analysis_parameters['steps']):
            approx_qubo = approximation_dict[str(i + 1)]['qubo']
            if isinstance(self.model.__class__, CombinedOneHotFeatureModel.__class__):
                approx_qubo = qubo - qubo_to_approx + approx_qubo
            #qubo_heatmap(approx_qubo)
            min_solution_quality, _, _, mean_solution_quality, *_ \
                = get_quality_of_approxed_qubo(qubo, approx_qubo, solutions, config)
            # print('step ', i)
            # print(min_solution_quality)
            stepwise_approx_quality.append(np.floor(1 - min_solution_quality))
            stepwise_min_approx_quality.append(min_solution_quality)
            stepwise_mean_approx_quality.append(mean_solution_quality)
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

