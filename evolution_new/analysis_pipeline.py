import numpy as np

from evolution_new.new_visualisation import visualisation_pipeline, visualize_boxplot_comparison, \
    visualize_boxplot_comparison_multiple_models
from model_analysis import ModelAnalysis


def get_models_dict(models_dict: dict) -> dict:
    model_analysis_dict = {}
    for training_cfg_name in models_dict:
        model_analysis_dict[training_cfg_name] = ModelAnalysis(training_cfg_name,
                                                               models_dict[training_cfg_name]['configs'],
                                                               models_dict[training_cfg_name]['analysis_parameters'])
        model_analysis_dict[training_cfg_name].run_analysis()
    return model_analysis_dict


def create_baseline_data_dict(baseline_data: list[list, list, list, list, list, list, list], steps,
                              sorted_approx=True, data_index=0, dotted=False, color='black') -> dict:
    return {
        'percent_list': baseline_data[1],
        'color': color,
        'dotted': dotted,
        'baseline_approx_data': baseline_data[data_index],
        'legend': f'stepwise-approximation: {steps} steps, '
                  f'{"smallest entries first" if sorted_approx else "random entries"}, '
                  f'{"mean" if data_index == 3 or data_index == 6 else "best"}'
    }


def get_visualisation_title(evaluation_type, config, solver, models=False):
    problems = config["pipeline"]["problems"]["problems"]
    title = f'{evaluation_type}, trained model{"s" if models else ""}, problem{"s" if len(problems) > 1 else ""}: '
    for problem in problems:
        title = title + problem + ', '
    title = title + f'Max-size: {config["pipeline"]["problems"]["qubo_size"]}, Solver: ' \
                    f'{solver}'
    return title


def get_model_config_description(model_name: str, model_analysis: ModelAnalysis, config_nr: int, kind: str) ->str:
    descr = model_name
    config = model_analysis.config_list[config_nr]
    problems = config["pipeline"]["problems"]["problems"]
    descr = descr + ', evaluated on: '
    for problem in problems:
        descr = descr + problem + ', '
    descr = descr + f'Max-size: {config["pipeline"]["problems"]["qubo_size"]}'
    if 'scale_list' in model_analysis.analysis_parameters:
        descr = descr + ', desired approximation: ' + str(model_analysis.analysis_parameters['scale_list'][config_nr])
    return descr + kind


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

    def visualize_baseline_correct_mean(self, analysis_dict: dict):
        visualisation_dict = {
            'baseline_data': [],
            'evaluation_results': [],
            'title': 'Model comparison',
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
                                                  color=model_dict['baseline_colors'][idx])
                    )
                    config_name_set.add(config_name)

                visualisation_dict['evaluation_results'].append(
                    {
                        'color': model_dict['colors'][idx],
                        'marker': 4,
                        'evol_y': [np.mean(model_analyis_results['approximation_quality_dict']
                                           ['solution_quality_list'])],
                        'evol_x': [np.mean(model_analyis_results['approximation_quality_dict']['approx_percent_list'])],
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
                            'evol_x': [np.mean(model_analyis_results['approximation_quality_dict']['approx_percent_list'])],
                            'label': get_model_config_description(model_dict['model_name'],
                                                                  model_analysis,
                                                                  config_nr,
                                                                  ', classical heuristic (without approximation)')
                        }
                    )
        visualisation_pipeline(visualisation_dict)

    def visualize_baseline_correct_incorrect(self, analysis_dict: dict):
        model_analysis = self.models_dict[analysis_dict['model']]
        model_analyis_results = model_analysis.model_result_list[analysis_dict['config']]
        visualisation_pipeline({
            'baseline_data': [create_baseline_data_dict(model_analyis_results['baseline'],
                                                        model_analysis.analysis_parameters['steps'])],
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
            'title': get_visualisation_title('Correct & incorrect solutions',
                                             model_analysis.config_list[analysis_dict['config']],
                                             model_analysis.analysis_parameters['solver']),
            'x_label': 'approximated qubo entries in percent',
            'y_label': '1 = correct solution found, 0 = correct solution not found'
        })

    def visualize_relative_quality_with_mean(self, analysis_dict: dict):
        model_analysis = self.models_dict[analysis_dict['model']]
        model_analyis_results = model_analysis.model_result_list[analysis_dict['config']]
        visualisation_pipeline({  # Sorted/random with min/mean pipeline
            'baseline_data': [
                create_baseline_data_dict(model_analyis_results['baseline'],
                                          model_analysis.analysis_parameters['steps'], data_index=3,
                                          color=analysis_dict['baseline_colors'][0]),
                create_baseline_data_dict(model_analyis_results['baseline'],
                                          model_analysis.analysis_parameters['steps'], data_index=4, dotted=True,
                                          color=analysis_dict['baseline_colors'][0]),
                create_baseline_data_dict(model_analyis_results['baseline'],
                                          model_analysis.analysis_parameters['steps'],
                                          data_index=5, sorted_approx=False, color=analysis_dict['baseline_colors'][1]),
                create_baseline_data_dict(model_analyis_results['baseline'],
                                          model_analysis.analysis_parameters['steps'], data_index=6,
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
            'title': get_visualisation_title('Solution quality of every solution',
                                             model_analysis.config_list[analysis_dict['config']],
                                             model_analysis.analysis_parameters['solver']),
            'x_label': 'approximated qubo entries in percent',
            'y_label': 'solution quality (min energy - energy) / min energy'
        })

    def visualize_boxplot_one_problem(self, analysis_dict: dict):
        model_analysis = self.models_dict[analysis_dict['model']]
        model_analyis_results = model_analysis.model_result_list[analysis_dict['config']]
        visualize_boxplot_comparison({
            'data_list':
                [
                    {
                        'min': model_analyis_results['approximation_quality_dict']['min_solution_quality_list'],
                        'mean': model_analyis_results['approximation_quality_dict']['mean_solution_quality_list'],
                        'tick_name': 'Model performance'
                    },
                    {
                        'min': model_analyis_results['approximation_quality_dict']['classical_min_solution_quality'],
                        'mean': model_analyis_results['approximation_quality_dict']['classical_mean_solution_quality'],
                        'tick_name': 'Classical algorithm'
                    },
                    {
                        'min': model_analyis_results['approximation_quality_dict']['random_min_solution_quality'],
                        'mean': model_analyis_results['approximation_quality_dict']['random_mean_solution_quality'],
                        'tick_name': 'Random solutions'
                    },
                    {
                        'min': model_analyis_results['approximation_quality_dict']['repeat_qubo_min_solution_quality'],
                        'mean': model_analyis_results['approximation_quality_dict']['repeat_qubo_mean_solution_quality'],
                        'tick_name': 'Original QUBO'
                    }
                ],
            'colors':
                {
                    'min': '#D7191C',
                    'mean': '#2C7BB6'
                },
            'y_label': 'solution quality (min energy - energy) / min energy',
            'title': get_visualisation_title('Comparison of different approaches',
                                             model_analysis.config_list[analysis_dict['config']],
                                             model_analysis.analysis_parameters['solver'])
        })

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
        visualize_boxplot_comparison({
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
            'title': 'Comparison between different models / evaluations'
        })


