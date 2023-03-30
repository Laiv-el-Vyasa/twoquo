import numpy as np

from approximation import get_approximated_qubos
from config import load_cfg
from evolution_new.combined_evolution_training import get_data_from_training_config
from evolution_new.evolution_utils import get_quality_of_approxed_qubo, get_qubo_approx_mask, get_file_name, \
    get_relative_size_of_approxed_entries
from evolution_new.pygad_learning import PygadLearner
from evolution_new.learning_model import LearningModel
from new_visualisation import visualize_evol_results, qubo_heatmap, visualize_two_result_points


solver = 'qbsolv_simulated_annealing'


class TrainingAnalysis:
    def __init__(self, config_name: str, analysis_parameters: dict):
        self.model, learning_parameters, fitness_func = get_data_from_training_config(config_name)
        self.pygad_learner = PygadLearner(self.model, learning_parameters, fitness_func)
        self.config = load_cfg(cfg_id=learning_parameters['config_name'])
        self.analysis_name = get_file_name(analysis_parameters['analysis_name'], self.config,
                                           learning_parameters['fitness_parameters'], analysis=True)
        self.config['pipeline']['problems']['n_problems'] *= 10
        self.analysis_parameters = analysis_parameters
        if not self.model.load_best_model(learning_parameters['training_name']):
            self.pygad_learner.save_best_model()
            self.model.load_best_model(learning_parameters['training_name'])

    def run_analysis(self):
        approximation_quality_dict = self.get_model_approximation_quality()
        analysis_baseline = self.get_analysis_baseline()
        visualize_evol_results(analysis_baseline[0], analysis_baseline[1],
                               (approx_percent_list, mean_solution_quality), self.analysis_name,
                               self.config["pipeline"]["problems"]["problems"][0],
                               self.config['pipeline']['problems']['qubo_size'], 'qbsolv_simulated_annealing',
                               self.analysis_parameters['steps'], boxplot=self.analysis_parameters['boxplot'])
        visualize_two_result_points(analysis_baseline[0], analysis_baseline[1],
                                    (correct_approx_list, mean_solution_quality),
                                    (incorrect_approx_list, mean_solution_quality),
                                    steps=self.analysis_parameters['steps'], baseline=True,
                                    title='Correct and incorrect approximations')
        visualize_two_result_points(analysis_baseline[0], analysis_baseline[1],
                                    (correct_approx_list, correct_approx_size),
                                    (incorrect_approx_list, incorrect_approx_size), baseline=False,
                                    title='Relative size of approximated entries')

    def get_model_approximation_quality(self) -> dict:
        return_dict = {
            'solution_quality_list': [],
            'approx_percent_list': [],
            'correct_approx_list': [],
            'incorrect_approx_list': [],
            'correct_approx_size': [],
            'incorrect_approx_size': []
        }
        problem_dict = self.model.get_approximation(self.model.get_training_dataset(self.config))
        approx_qubo_list, solutions_list, qubo_list = problem_dict['approxed_qubo_list'], \
                                                      problem_dict['solutions_list'], problem_dict['qubo_list']
        for idx, (qubo, approx_qubo, solutions) in enumerate(zip(qubo_list, approx_qubo_list, solutions_list)):
            print(f'Approximating problem {idx} via model')
            if idx < self.analysis_parameters['show_qubo_mask']:
                qubo_heatmap(qubo)
                qubo_heatmap(get_qubo_approx_mask(approx_qubo, qubo))
            min_solution_quality, _, approx_percent = get_quality_of_approxed_qubo(qubo, approx_qubo,
                                                                                   solutions, self.config)
            return_dict['solution_quality_list'].append((np.floor(1 - min_solution_quality)))
            approx_size = get_relative_size_of_approxed_entries(approx_qubo, qubo)
            print('approx size: ', approx_size)
            print(min_solution_quality, approx_percent)
            if min_solution_quality <= 0 and approx_percent != 0:
                print('True solution found')
                return_dict['correct_approx_list'].append(approx_percent)
                return_dict['correct_approx_size'].append(approx_size)
            else:
                return_dict['incorrect_approx_list'].append(approx_percent)
                return_dict['incorrect_approx_size'].append(approx_size)
            return_dict['approx_percent_list'].append(approx_percent)
        return return_dict

    def get_analysis_baseline(self) -> list[list, list]:
        try:
            analysis_baseline = np.load(f'analysis_baseline/{self.analysis_name}.npy')
            print('Analysis baseline loaded')
        except FileNotFoundError:
            analysis_baseline = self.get_new_analysis_baseline()
            np.save(f'analysis_baseline/{self.analysis_name}', analysis_baseline)
        print(analysis_baseline)
        return analysis_baseline

    def get_new_analysis_baseline(self) -> list[list, list]:
        analysis_baseline = [[], []]
        problem_dict = self.model.get_training_dataset(self.config)
        stepwise_approx_quality = self.get_stepwise_approx_quality(problem_dict)
        # Prepare array for saving and display
        analysis_baseline[0] = stepwise_approx_quality
        step_list = [n / (self.analysis_parameters['steps'] + 1) for n in range(self.analysis_parameters['steps'] + 1)]
        analysis_baseline[1] = step_list
        return analysis_baseline

    def get_stepwise_approx_quality(self, problem_dict: dict) -> list:
        qubo_list, solutions_list = problem_dict['qubo_list'], problem_dict['solutions_list']
        solution_quality_list = []
        for idx, (qubo, solutions) in enumerate(zip(qubo_list, solutions_list)):
            print(f'Approximating problem {idx} for baseline')
            solution_quality_list.append(self.get_stepwise_approx_quality_for_qubo(qubo, solutions))
        return self.rotate_solution_quality_list(solution_quality_list)

    def get_stepwise_approx_quality_for_qubo(self, qubo: list, solutions: list) -> list:
        stepwise_approx_quality = [1.]
        approximation_dict, _ = get_approximated_qubos(qubo, False, True, self.analysis_parameters['steps'],
                                                       sorted_approx=self.analysis_parameters['sorted'])
        for i in range(self.analysis_parameters['steps']):
            approx_qubo = approximation_dict[str(i + 1)]['qubo']
            min_solution_quality, *_ = get_quality_of_approxed_qubo(qubo, approx_qubo, solutions, self.config)
            stepwise_approx_quality.append(np.floor(1 - min_solution_quality))
        return stepwise_approx_quality

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
