from matplotlib import pyplot
import matplotlib.lines as lines
import numpy as np
from config import load_cfg

color_dict = {
    'NP': 'green',
    "MC": "blue",
    "M2SAT": "red"
}

cfg = load_cfg(cfg_id='test')
problem_name_array = cfg['pipeline']['problems']['problems']


def qubo_heatmap(qubo_matrix):
    fig, ax = pyplot.subplots()
    mesh = ax.pcolormesh(qubo_matrix, cmap="plasma")
    ax.set_xticks([i for i in range(qubo_matrix.shape[0])], minor=True)
    ax.set_yticks([i for i in range(qubo_matrix.shape[0])], minor=True)
    ax.grid(True, color="black", which="both", linewidth=0.1, linestyle="-")
    ax.set_title("QUBO Heatmap")
    fig.colorbar(mesh, ax=ax)
    pyplot.gca().invert_yaxis()
    pyplot.show()


def approx_quality_score_graphs_problems(overall_data, approx_fixed_number, include_zero=False):
    fig, ax = pyplot.subplots()
    percentage_steps = overall_data['approximation_steps']
    if include_zero:
        approximation_steps = [0]
    else:
        approximation_steps = []
    approximation_steps.extend(percentage_steps)
    legend_lines = []
    for key in overall_data:
        if key != 'approximation_steps' and key != 'solver':
            mean_energy_list, mean_score_list = aggregate_problem_data(overall_data[key], include_zero)
            print(mean_energy_list)
            print(approximation_steps)
            ax.plot(approximation_steps, mean_energy_list, color=color_dict[key], markersize=8)
            legend_lines.append(lines.Line2D([], [], color=color_dict[key],
                                     markersize=8, label=f'{key} energy'))
            ax.plot(approximation_steps, mean_score_list, color=color_dict[key], linestyle='dashed')
            legend_lines.append(lines.Line2D([], [], color=color_dict[key], linestyle='dashed',
                                     markersize=8, label=f'{key} score'))
    ax.legend(handles=legend_lines)
    ax.set_ylabel('quality')
    if approx_fixed_number:
        label_string = 'entries'
    else:
        label_string = 'values'
    ax.set_xlabel(f'approximated {label_string} in percent')
    ax.set_title('Solver: ' + overall_data['solver'])
    pyplot.show()


def aggregate_problem_data(problem_data_list, include_zero):
    score_list = []
    energy_list = []
    for problem_data_dict in problem_data_list:
        score = []
        if include_zero:
            score.append(1)
        energy = []
        if include_zero:
            energy.append(1)
        for approx_step_data in problem_data_dict['approximation']:
            energy.append(approx_step_data['rel_energy'])
            score.append(approx_step_data['rel_score'])
        score_list.append(score)
        energy_list.append(energy)
    #print(energy_list)
    return np.mean(energy_list, axis=0), np.mean(score_list, axis=0)


def approx_quality_graphs_problems_ml(aggregated_approx_data, percent_data, learned_data_points, approx_fixed_number, solver):
    fig, ax = pyplot.subplots()
    legend_lines = []
    for problem_number, problem_approx_data in enumerate(aggregated_approx_data):
        color = color_dict[problem_name_array[problem_number]]
        ax.plot(percent_data, problem_approx_data, color=color, markersize=8)
        legend_lines.append(lines.Line2D([], [], color=color,
                                             markersize=8, label=f'{problem_name_array[problem_number]} solution quality'))
        ax.plot(learned_data_points[problem_number][0], learned_data_points[problem_number][1],
                color=color, marker='x', markersize=12)
        legend_lines.append(lines.Line2D([], [], color=color, linestyle='dashed',
                                             markersize=12, marker='x', label=f'{problem_name_array[problem_number]} learned approximation'))
    ax.legend(handles=legend_lines)
    ax.set_ylabel('solution quality')
    if approx_fixed_number:
        label_string = 'entries'
    else:
        label_string = 'values'
    ax.set_xlabel(f'approximated {label_string} in percent')
    ax.set_title('Solver: ' + solver)
    pyplot.show()
