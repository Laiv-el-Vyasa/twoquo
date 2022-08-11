from matplotlib import pyplot
import matplotlib.lines as lines
import numpy as np

color_dict = {
    'NP': 'green',
    "MC": "blue",
    "M2SAT": "red"
}

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


def approx_quality_graphs(overall_data, approx_fixed_number, include_zero=False):
    fig, ax = pyplot.subplots()
    percentage_steps = overall_data['approximation_steps']
    if include_zero:
        approximation_steps = [0]
    else:
        approximation_steps = []
    approximation_steps.extend(percentage_steps)
    legend_lines = []
    for key in overall_data:
        if key != 'approximation_steps':
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

