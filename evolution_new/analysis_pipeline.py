from evolution_new.new_visualisation import visualisation_pipeline
from model_analysis import ModelAnalysis


def get_models_dict(models_dict: dict) -> dict:
    for training_cfg_name in models_dict:
        models_dict[training_cfg_name]['model_analysis'] = ModelAnalysis(training_cfg_name,
                                                                         models_dict[training_cfg_name]['configs'],
                                                                         models_dict[training_cfg_name]
                                                                         ['analysis_paramaters'])
        models_dict['model_analysis'].run_analysis()
    return models_dict


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
        pass

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
                    'color': analysis_dict['colors'][0],
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
        pass

    def visualize_boxplot_one_problem(self, analysis_dict: dict):
        pass

