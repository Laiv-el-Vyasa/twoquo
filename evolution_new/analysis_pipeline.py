import numpy as np

from evolution_new.new_visualisation import visualisation_pipeline, visualize_boxplot_comparison, \
    visualize_boxplot_comparison_multiple_models, visualize_boxplot_comparison_quantum
from model_analysis import ModelAnalysis

# Executing the analysis given in an analysis-dict


# Get all analysis-results for the supplied models and collect them in a dict
def get_models_dict(models_dict: dict) -> dict:
    model_analysis_dict = {}
    for training_cfg_name in models_dict:
        model_analysis_dict[training_cfg_name] = ModelAnalysis(training_cfg_name,
                                                               models_dict[training_cfg_name]['configs'],
                                                               models_dict[training_cfg_name]['analysis_parameters'])
        model_analysis_dict[training_cfg_name].run_analysis()
    return model_analysis_dict


# Create a dict for baseline visualisation
def create_baseline_data_dict(baseline_data: list[list, list, list, list, list, list, list], steps,
                              model_analysis: ModelAnalysis, config_nr: int,
                              sorted_approx=True, data_index=0, dotted=False, color='black') -> dict:
    config = model_analysis.config_list[config_nr]
    problems = config["pipeline"]["problems"]["problems"]
    config_description = ''
    for problem in problems:
        config_description = config_description + problem + ', '
    config_description = config_description + str(config["pipeline"]["problems"]["qubo_size"])
    return {
        'percent_list': baseline_data[1],
        'color': color,
        'dotted': dotted,
        'baseline_approx_data': baseline_data[data_index],
        'legend': f'{config_description}, '
                  f'stepwise-approximation: {steps} steps, '
                  f'{"smallest entries first" if sorted_approx else "random entries"}, '
                  f'{"mean" if data_index == 3 or data_index == 6 else "best solution"}'
    }


# Title for the plots
def get_visualisation_title(evaluation_type, config, solver, models=False):
    problems = config["pipeline"]["problems"]["problems"]
    title = f'{evaluation_type},\nevaluation on problem{"s" if len(problems) > 1 else ""}: '
    for problem in problems:
        title = title + problem + ', '
    title = title + f'Max-size: {config["pipeline"]["problems"]["qubo_size"]}'  # \
    #                f', Solver: ' f'{solver}'
    return title


# Description of the models
def get_model_config_description(model_name: str, model_analysis: ModelAnalysis, config_nr: int, kind: str) ->str:
    descr = model_name
    config = model_analysis.config_list[config_nr]
    problems = config["pipeline"]["problems"]["problems"]
    descr = descr + ', \nevaluated on: '
    for problem in problems:
        descr = descr + problem + ', '
    descr = descr + f'{config["pipeline"]["problems"]["qubo_size"]}'
    if 'scale_list' in model_analysis.analysis_parameters:
        descr = descr + ', \ndesired approximation: ' + str(model_analysis.analysis_parameters['scale_list'][config_nr])
    return descr + kind


# Derive the error-bars for the respective visualisation
# Calculate the standard deviation for alle entries above and below the mean respectively
# (Else we can get error-bars extending beyond 0 and 1)
def create_upper_lower_errorbars(approx_percent_list: list) -> list[list, list]:
    mean = np.mean(approx_percent_list)
    list_above = [x - mean for x in approx_percent_list if x > mean]
    list_below = [x - mean for x in approx_percent_list if x < mean]
    return [
        [np.sqrt(np.sum(np.square(list_below)) / len(list_below))],
        [np.sqrt(np.sum(np.square(list_above)) / len(list_above))]
    ]


class AnalysisPipeline:
    def __init__(self, analysis_dict: dict):
        self.analysis_dict = analysis_dict
        self.models_dict = get_models_dict(analysis_dict['models'])

    def start_visualisation(self):
        for analysis in self.analysis_dict['analysis']:
            if analysis['type'] == 'baseline_correct_mean':
                self.visualize_baseline_correct_mean(analysis)
            elif analysis['type'] == 'baseline_correct_incorrect':
                self.visualize_baseline_correct_incorrect(analysis)
            elif analysis['type'] == 'relative_quality_with_mean':
                self.visualize_relative_quality_with_mean(analysis)
            elif analysis['type'] == 'boxplot_one':
                self.visualize_boxplot_one_problem(analysis)
            elif analysis['type'] == 'boxplot_multiple':
                self.visualize_boxplot_multiple_problems(analysis)
            elif analysis['type'] == 'boxplot_quantum':
                self.visualize_quantum_boxplots(analysis)

    # Compare the percentage of correct solutions between models and with teh respective baselines
    def visualize_baseline_correct_mean(self, analysis_dict: dict):
        visualisation_dict = {
            'baseline_data': [],
            'evaluation_results': [],
            'title': f'Comparison with approximation baseline, different '
                     f'{"models / " if len(analysis_dict["models"]) > 1 else ""}evaluations',
            'x_label': 'approximated qubo entries in percent',
            'y_label': 'percentage of correct solutions found'
        }
        config_name_set = set()
        for model_name in analysis_dict['models']:
            model_analysis = self.models_dict[model_name]
            model_dict = analysis_dict['models'][model_name]
            for idx, config_nr in enumerate(model_dict['configs']):
                model_analyis_results = model_analysis.model_result_list[config_nr]

                config_name = self.analysis_dict['models'][model_name]['configs'][config_nr]
                if config_name not in config_name_set:
                    visualisation_dict['baseline_data'].append(
                        create_baseline_data_dict(model_analyis_results['baseline'],
                                                  model_analysis.analysis_parameters['steps'],
                                                  model_analysis,
                                                  config_nr, dotted=True if config_nr == 1 else False,
                                                  color=model_dict['baseline_colors'][idx])
                    )
                    config_name_set.add(config_name)

                visualisation_dict['evaluation_results'].append(
                    {
                        'color': model_dict['colors'][idx],
                        'marker': 4 + config_nr if not analysis_dict['compare'] * 2 else 4,
                        'evol_y': [np.mean(model_analyis_results['approximation_quality_dict']
                                           ['solution_quality_list'])],
                        'evol_x': [np.mean(model_analyis_results['approximation_quality_dict']['approx_percent_list'])],
                        'evol_x_err': [create_upper_lower_errorbars(model_analyis_results['approximation_quality_dict']
                                                                    ['approx_percent_list'])],
                        'std_dev': [np.std(model_analyis_results['approximation_quality_dict']['approx_percent_list'])],
                        'label': get_model_config_description(model_dict['model_name'],
                                                              model_analysis,
                                                              config_nr,
                                                              ', model performance' if analysis_dict['compare'] else
                                                              '')
                    }
                )
                if analysis_dict['compare']:
                    visualisation_dict['evaluation_results'].append(
                        {
                            'color': model_dict['colors'][idx],
                            'marker': 5,
                            'evol_y': [np.mean(model_analyis_results['approximation_quality_dict']
                                               ['random_solution_quality'])],
                            'evol_x': [np.mean(model_analyis_results['approximation_quality_dict']['approx_percent_list'])],
                            'label': get_model_config_description(model_dict['model_name'],
                                                                  model_analysis,
                                                                  config_nr,
                                                                  ', random solutions (without approximation)')
                        }
                    )
                    visualisation_dict['evaluation_results'].append(
                        {
                            'color': model_dict['colors'][idx],
                            'marker': 6,
                            'evol_y': [np.mean(model_analyis_results['approximation_quality_dict']
                                               ['classical_solution_quality'])],
                            'evol_x': [np.mean(model_analyis_results['approximation_quality_dict']
                                               ['approx_percent_list'])],
                            'label': get_model_config_description(model_dict['model_name'],
                                                                  model_analysis,
                                                                  config_nr,
                                                                  ', classical heuristic (without approximation)')
                        }
                    )
        visualisation_pipeline(visualisation_dict)

    # Add all solutions with their approximation percentage as a point, 0 for incorrect, 1 for correct solutions
    # Baseline is present
    def visualize_baseline_correct_incorrect(self, analysis_dict: dict):
        model_analysis = self.models_dict[analysis_dict['model']]
        model_analyis_results = model_analysis.model_result_list[analysis_dict['config']]
        visualisation_pipeline({
            'baseline_data': [create_baseline_data_dict(model_analyis_results['baseline'],
                                                        model_analysis.analysis_parameters['steps'],
                                                        model_analysis, 0)],
            'evaluation_results': [
                {
                    'color': analysis_dict['colors'][0],
                    'marker': 4,
                    'evol_y': [1 for _ in model_analyis_results['approximation_quality_dict']['correct_approx_list']],
                    'evol_x': model_analyis_results['approximation_quality_dict']['correct_approx_list'],
                    'label': 'Correct solutions suggested by model'
                },
                {
                    'color': analysis_dict['colors'][1],
                    'marker': 4,
                    'evol_y': [0 for _ in model_analyis_results['approximation_quality_dict']['incorrect_approx_list']],
                    'evol_x': model_analyis_results['approximation_quality_dict']['incorrect_approx_list'],
                    'label': 'Incorrect solutions suggested by model'
                },
            ],
            'title': get_visualisation_title('Correct & incorrect solutions, ',
                                             model_analysis.config_list[analysis_dict['config']],
                                             model_analysis.analysis_parameters['solver']),
            'x_label': 'approximated qubo entries in percent',
            'y_label': '1 = correct solution found, 0 = correct solution not found'
        })

    # Add points for min and mean solution quality on the y-axis and the approx percentage on the x-axis
    # Together with baseline
    def visualize_relative_quality_with_mean(self, analysis_dict: dict):
        model_analysis = self.models_dict[analysis_dict['model']]
        model_analyis_results = model_analysis.model_result_list[analysis_dict['config']]
        visualisation_pipeline({  # Sorted/random with min/mean pipeline
            'baseline_data': [
                create_baseline_data_dict(model_analyis_results['baseline'],
                                          model_analysis.analysis_parameters['steps'],
                                          model_analysis, 0, data_index=3,
                                          color=analysis_dict['baseline_colors'][0]),
                create_baseline_data_dict(model_analyis_results['baseline'],
                                          model_analysis.analysis_parameters['steps'],
                                          model_analysis, 0, data_index=4, dotted=True,
                                          color=analysis_dict['baseline_colors'][0]),
                create_baseline_data_dict(model_analyis_results['baseline'],
                                          model_analysis.analysis_parameters['steps'],
                                          model_analysis, 0,
                                          data_index=5, sorted_approx=False, color=analysis_dict['baseline_colors'][1]),
                create_baseline_data_dict(model_analyis_results['baseline'],
                                          model_analysis.analysis_parameters['steps'],
                                          model_analysis, 0, data_index=6,
                                          sorted_approx=False, dotted=True, color=analysis_dict['baseline_colors'][1])
            ],
            'evaluation_results': [
                {
                    'color': 'blue',
                    'marker': 4,
                    'alpha': 0.7,
                    'evol_y': model_analyis_results['approximation_quality_dict']['min_solution_quality_list'],
                    'evol_x': model_analyis_results['approximation_quality_dict']['approx_percent_list'],
                    'label': 'Min solution quality of approximation suggested by model'
                },
                {
                    'color': 'slateblue',
                    'marker': 5,
                    'alpha': 0.7,
                    'evol_y': model_analyis_results['approximation_quality_dict']['mean_solution_quality_list'],
                    'evol_x': model_analyis_results['approximation_quality_dict']['approx_percent_list'],
                    'label': 'Mean solution quality of approximation suggested by model'
                },
            ],
            'title': get_visualisation_title('Solution quality of every solution, ',
                                             model_analysis.config_list[analysis_dict['config']],
                                             model_analysis.analysis_parameters['solver']),
            'x_label': 'approximated qubo entries in percent',
            'y_label': 'solution quality (min energy - energy) / min energy'
        })

    # Name misleading, boxplot comparison of different approaches, comparison of different models possible
    def visualize_boxplot_one_problem(self, analysis_dict: dict):
        model_analysis = self.models_dict[analysis_dict['model']]
        analysis_list = []
        for config_nr in analysis_dict['config_list']:
            model_analyis_results = model_analysis.model_result_list[config_nr]
            analysis_list.append({
                'data_list':
                    [
                        {
                            'min': model_analyis_results['approximation_quality_dict']['repeat_qubo_min_solution_quality'],
                            'mean': model_analyis_results['approximation_quality_dict']['repeat_qubo_mean_solution_quality'],
                            'tick_name': 'Original\nQUBO'
                        },
                        {
                            'min': model_analyis_results['approximation_quality_dict']['min_solution_quality_list'],
                            'mean': model_analyis_results['approximation_quality_dict']['mean_solution_quality_list'],
                            'tick_name': f'Model\napprox'
                                         #f',\n{analysis_dict["model_name"]}'
                        },
                        {
                            'min': model_analyis_results['approximation_quality_dict']
                                                        ['stepwise_approx_min_solution_quality'],
                            'mean': model_analyis_results['approximation_quality_dict']
                                                         ['stepwise_approx_mean_solution_quality'],
                            'tick_name': f'Stepwise\napprox'
                        },
                        {
                            'min': model_analyis_results['approximation_quality_dict']['classical_min_solution_quality'],
                            'mean': model_analyis_results['approximation_quality_dict']['classical_mean_solution_quality'],
                            'tick_name': 'Heuristic\nalgorithm'
                        },
                        {
                            'min': model_analyis_results['approximation_quality_dict']['random_min_solution_quality'],
                            'mean': model_analyis_results['approximation_quality_dict']['random_mean_solution_quality'],
                            'tick_name': 'Random\nsolutions'
                        }
                    ],
                'colors':
                    {
                        'min': '#D7191C',
                        'mean': '#2C7BB6'
                    },
                'y_label': 'solution quality (min energy - energy) / min energy',
                'title': get_visualisation_title('Comparison of different approaches',
                                                 model_analysis.config_list[config_nr],
                                                 model_analysis.analysis_parameters['solver'])
            })
        visualize_boxplot_comparison(analysis_list)

    # Boxplot comparison of different approaches, no longer in use
    def visualize_boxplot_multiple_problems(self, analysis_dict: dict):
        data_list = []
        for model_name in analysis_dict['models']:
            model_analysis = self.models_dict[model_name]
            model_dict = analysis_dict['models'][model_name]
            for idx, config_nr in enumerate(model_dict['configs']):
                model_analyis_results = model_analysis.model_result_list[config_nr]
                data_list.append({
                    'model': model_analyis_results['approximation_quality_dict']['min_solution_quality_list'],
                    'classical': model_analyis_results['approximation_quality_dict']
                    ['classical_min_solution_quality'],
                    'random': model_analyis_results['approximation_quality_dict']['random_min_solution_quality'],
                    'qubo': model_analyis_results['approximation_quality_dict']['repeat_qubo_min_solution_quality'],
                    'tick_name': get_model_config_description(model_dict['model_name'], model_analysis, config_nr, '')
                })
        visualize_boxplot_comparison_multiple_models({
            'data_list':
                data_list,
            'colors':
                {
                    'model': '#D7191C',
                    'classical': '#2C7BB6',
                    'random': '#FA9C1B',
                    'qubo': '#9400D3'
                },
            'y_label': 'solution quality (min energy - energy) / min energy',
            'title': f'Comparison of solution quality between different '
                     f'{"models / " if len(analysis_dict["models"]) > 1 else ""}evaluations'
        })

    # Quantum visualisation,including a comparison of solution quality and embedding size
    def visualize_quantum_boxplots(self, analysis_dict: dict):
        model_analysis = self.models_dict[analysis_dict['model']]
        config_list = analysis_dict['config_list']
        visualisation_list_quality = []
        visualisation_list_embedding = []
        for config_nr in config_list:
            model_analyis_results = model_analysis.model_result_list[config_nr]
            visualisation_list_quality.append({
                'data_list':
                    [
                        {
                            'min': model_analyis_results['quantum_quality_dict']['min_solution_quality_list_original'],
                            'mean': model_analyis_results['quantum_quality_dict']['mean_solution_quality_list_original'],
                            'tick_name': f'Original QUBO'
                        },
                        {
                            'min': model_analyis_results['quantum_quality_dict']['min_solution_quality_list_approx'],
                            'mean': model_analyis_results['quantum_quality_dict']['mean_solution_quality_list_approx'],
                            'tick_name': f'Approximated QUBO'
                        },
                    ],
                'colors':
                    {
                        'min': '#D7191C',
                        'mean': '#2C7BB6'
                    },
                'y_label': 'solution quality (min energy - energy) / min energy',
                'title': get_visualisation_title(f'Comparison of solution quality\non '
                                                 f'{model_analysis.analysis_parameters["quantum"]["qpu_name"]}',
                                                 model_analysis.config_list[config_nr],
                                                 '')
            })

            visualisation_list_embedding.append({
                'data_list':
                    [
                        {
                            'relative embedding size': model_analyis_results['quantum_quality_dict']
                                                        ['embedding_size_list_original'],
                            'avg chain length': model_analyis_results['quantum_quality_dict']
                                                         ['embedding_avg_chain_list_original'],
                            'tick_name': f'Original QUBO'
                        },
                        {
                            'relative embedding size': model_analyis_results['quantum_quality_dict']
                                                        ['embedding_size_list_approx'],
                            'avg chain length': model_analyis_results['quantum_quality_dict']
                                                         ['embedding_avg_chain_list_approx'],
                            'tick_name': f'Approximated QUBO'
                        },
                    ],
                'colors':
                    {
                        'relative embedding size': '#D7191C',
                        'avg chain length': '#2C7BB6'
                    },
                'embedding': True,
                'y_label': 'relative embedding size:\nphysical qubits / logical qubits',
                'y_label_2': 'average chain length',
                'title': get_visualisation_title(f'Comparison of embedding parameters\non '
                                                 f'{model_analysis.analysis_parameters["quantum"]["qpu_name"]}\n'
                                                 f'with '
                                                 f'{model_analysis.analysis_parameters["quantum"]["embedding_structure"]} '
                                                 f'embedding',
                                                 model_analysis.config_list[config_nr],
                                                 '')
            })
        visualize_boxplot_comparison(visualisation_list_quality)
        visualize_boxplot_comparison(visualisation_list_embedding)


