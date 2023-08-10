import matplotlib.figure
from matplotlib import pyplot
import matplotlib.lines as lines
import numpy as np
from config import load_cfg
import itertools


# Collection of methods for visualisation

# Plot the results together with the baseline
def visualisation_pipeline(evaluation_dict: dict):
    fig, ax = pyplot.subplots()
    legend_lines = []
    if 'baseline_data' in evaluation_dict:
        for baseline_data in evaluation_dict['baseline_data']:
            ax, legend_lines = add_baseline(ax, baseline_data, legend_lines)

    ax, legend_lines = plot_points(ax, evaluation_dict['evaluation_results'], legend_lines)
    ax.legend(handles=legend_lines, prop={'size': 16})
    if 'scale_axis' in evaluation_dict:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    ax.set_ylabel(evaluation_dict['y_label'], fontsize=16)
    ax.set_xlabel(evaluation_dict['x_label'], fontsize=16)
    ax.set_title(evaluation_dict['title'], fontsize=24)
    pyplot.xticks(fontsize=14)
    pyplot.yticks(fontsize=14)
    matplotlib.pyplot.subplots_adjust(left=0.05, bottom=0.075, right=0.95, top=0.925, wspace=None, hspace=None)
    pyplot.show()


def add_baseline(ax: matplotlib.axes.Axes, baseline_data: dict, legend_lines: list) -> tuple:
    color = baseline_data['color']
    if baseline_data['dotted']:
        linestyle = 'dashed'
    else:
        linestyle = 'solid'
    ax.plot(baseline_data['percent_list'], baseline_data['baseline_approx_data'], color=color, linestyle=linestyle,
            markersize=8)
    legend_lines.append(lines.Line2D([], [], color=color, markersize=8, linestyle=linestyle,
                                     label=baseline_data['legend']))
    return ax, legend_lines


def plot_points(ax: matplotlib.axes.Axes, evaluation_results: dict, legend_lines: list) -> tuple:
    y_list = []
    for evol_result in evaluation_results:
        color = evol_result['color']
        marker = evol_result['marker']
        if 'alpha' in evol_result:
            alpha = evol_result['alpha']
        else:
            alpha = 1
        if 'evol_x_err' in evol_result:
            print(evol_result)
            for x, y, x_err in zip(evol_result['evol_x'], evol_result['evol_y'], evol_result['evol_x_err'],):
                if y in y_list:
                    y -= 0.01
                ax.errorbar(x, y, xerr=x_err, capsize=6, color=color, marker=(marker, 2), markersize=12, alpha=alpha)
                y_list.append(y)
        elif 'std_dev' in evol_result:
            print(evol_result)
            for x, y, x_err in zip(evol_result['evol_x'], evol_result['evol_y'], evol_result['std_dev'],):
                if y in y_list:
                    y -= 0.01
                ax.errorbar(x, y, xerr=x_err, capsize=6, color=color, marker=(marker, 2), markersize=12, alpha=alpha)
                y_list.append(y)
        else:
            for x, y in zip(evol_result['evol_x'], evol_result['evol_y']):
                ax.plot(x, y, color=color, marker=(marker, 2), markersize=12, alpha=alpha)
        legend_lines.append(lines.Line2D([], [], color=color, linestyle='None',
                                         markersize=12, marker=(marker, 2), label=evol_result['label']))
    return ax, legend_lines


# Compare the results of different approaches using boxplots for min and mean solution quality
# The results for multiple evaluated models can be displayed side by side
def visualize_boxplot_comparison(boxplot_data_list: list[dict]):
    fig, axes = pyplot.subplots(ncols=2)
    extreme_values = [[], []]
    for i in range(2):
        boxplot_data = boxplot_data_list[i]
        ax = axes[i]
        data_list = boxplot_data['data_list']
        color_dict = boxplot_data['colors']
        color_dict_keys = list(color_dict)
        tick_labels = []
        for i, data in enumerate(data_list):
            extreme_values[0].append(np.max(data[color_dict_keys[0]]))
            extreme_values[1].append(np.min(data[color_dict_keys[0]]))
            extreme_values[0].append(np.max(data[color_dict_keys[1]]))
            extreme_values[1].append(np.min(data[color_dict_keys[1]]))
            bxplt = ax.boxplot(data[color_dict_keys[0]], positions=[i * 2 - 0.3], widths=0.4)
            set_box_color(bxplt, color_dict[color_dict_keys[0]])
            bxplt = ax.boxplot(data[color_dict_keys[1]], positions=[i * 2 + 0.3], widths=0.4)
            set_box_color(bxplt, color_dict[color_dict_keys[1]])
            tick_labels.append(data['tick_name'])

        for analysis_name in color_dict:
            ax.plot([], c=color_dict[analysis_name], label=analysis_name)
        ax.legend(prop={'size': 16})

        ax.set_xticks(range(0, len(tick_labels) * 2, 2), tick_labels, fontsize=16)
        ax.set_xlim(-1, len(tick_labels)*2 - 1)
        ax.set_ylabel(boxplot_data['y_label'], fontsize=16)
        if 'embedding' in boxplot_data:
            ax.yaxis.label.set_color(color_dict['relative embedding size'])
            secax = ax.secondary_yaxis('right')
            secax.set_ylabel(boxplot_data['y_label_2'], fontsize=16)
            secax.yaxis.label.set_color(color_dict['avg chain length'])
            secax.tick_params(labelsize=14)
        ax.set_title(boxplot_data['title'], fontsize=24)
        pyplot.yticks(fontsize=30)
    for j in range(2):
        ax = axes[j]
        ax.set_ylim(np.min(extreme_values[1]) - 0.01, np.max(extreme_values[0]) + 0.01)
        #a = ax.flatten()
        ax.tick_params(axis='y', which='major', labelsize=14)
        ax.tick_params(axis='y', which='minor', labelsize=14)
        #ax.yticks(fontsize=14)
    matplotlib.pyplot.subplots_adjust(left=0.065, bottom=0.05, right=0.935, top=0.84, wspace=0.29, hspace=None)

    pyplot.show()


# Similar to the method above, designed to compare the embedding parameters
def visualize_boxplot_comparison_quantum(boxplot_data_list: list[list[dict], list[dict]]):
    fig, axes = pyplot.subplots(ncols=2, nrows=2)
    for i in range(2):
        ax_row = axes[i]
        for j in range(2):
            print(i, j)
            ax = ax_row[j]
            boxplot_data = boxplot_data_list[j][i]
            print(boxplot_data_list)
            data_list = boxplot_data['data_list']
            color_dict = boxplot_data['colors']
            color_dict_keys = list(color_dict)
            tick_labels = []
            for k, data in enumerate(data_list):
                bxplt = ax.boxplot(data[color_dict_keys[0]], positions=[k * 2 - 0.3], widths=0.4)
                set_box_color(bxplt, color_dict[color_dict_keys[0]])
                bxplt = ax.boxplot(data[color_dict_keys[1]], positions=[k * 2 + 0.3], widths=0.4)
                set_box_color(bxplt, color_dict[color_dict_keys[1]])
                tick_labels.append(data['tick_name'])

            for analysis_name in color_dict:
                ax.plot([], c=color_dict[analysis_name], label=analysis_name)
            ax.legend()

            ax.set_xticks(range(0, len(tick_labels) * 2, 2), tick_labels)
            ax.set_xlim(-1, len(tick_labels)*2 - 1)
            ax.set_ylabel(boxplot_data['y_label'])
            #ax.set_ylim(-0.03, 0.75)
            if 'embedding' in boxplot_data:
                ax.yaxis.label.set_color(color_dict['relative embedding size'])
                secax = ax.secondary_yaxis('right')
                secax.set_ylabel(boxplot_data['y_label_2'])
                secax.yaxis.label.set_color(color_dict['avg chain length'])
            ax.set_title(boxplot_data['title'])
    matplotlib.pyplot.subplots_adjust(left=0.04, bottom=0.08, right=0.96, top=0.92, wspace=0.2, hspace=0.4)
    pyplot.show()


# Deprecated method, compare different approaches as above, but without the mean solution quality
def visualize_boxplot_comparison_multiple_models(boxplot_data: dict):
    label_dict = {
        'model': 'Approximated QUBO',
        'classical': 'Heuristic Algorithm',
        'random': 'Random solution',
        'qubo': 'Original QUBO'
    }
    fig, ax = pyplot.subplots()
    data_list = boxplot_data['data_list']
    color_dict = boxplot_data['colors']
    tick_labels = []
    for i, data in enumerate(data_list):
        bxplt = pyplot.boxplot(data['model'], positions=[i * 3 - 0.3], widths=0.4)
        set_box_color(bxplt, color_dict['model'])
        bxplt = pyplot.boxplot(data['classical'], positions=[i * 3 + 0.3], widths=0.4)
        set_box_color(bxplt, color_dict['classical'])
        bxplt = pyplot.boxplot(data['random'], positions=[i * 3 - 0.9], widths=0.4)
        set_box_color(bxplt, color_dict['random'])
        bxplt = pyplot.boxplot(data['qubo'], positions=[i * 3 + 0.9], widths=0.4)
        set_box_color(bxplt, color_dict['qubo'])
        tick_labels.append(data['tick_name'])

    for analysis_name in color_dict:
        pyplot.plot([], c=color_dict[analysis_name], label=label_dict[analysis_name])
    pyplot.legend()
    pyplot.legend()

    pyplot.xticks(range(0, len(tick_labels) * 3, 3), tick_labels)
    pyplot.xlim(-2, len(tick_labels) * 3)
    #pyplot.yscale('symlog')
    ax.set_ylabel(boxplot_data['y_label'])
    ax.set_title(boxplot_data['title'])
    matplotlib.pyplot.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=None, hspace=None)
    pyplot.show()


def set_box_color(bp, color):
    pyplot.setp(bp['boxes'], color=color)
    pyplot.setp(bp['whiskers'], color=color)
    pyplot.setp(bp['caps'], color=color)
    pyplot.setp(bp['medians'], color=color)


def visualize_two_result_points(baseline_approx_data, percent_list, evol_results_1, evol_results_2,
                                steps=0, baseline=False, title=''):
    fig, ax = pyplot.subplots()
    legend_lines = []
    if baseline:
        color = 'black'
        ax.plot(percent_list, baseline_approx_data, color=color, markersize=8)
        legend_lines.append(lines.Line2D([], [], color=color, markersize=8, label=f'step-wise approximation, {steps} steps'))

    marker_size = 4
    color = 'green'
    print(evol_results_1)
    evol_x_1, evol_y_1 = evol_results_1
    if isinstance(evol_y_1, list):
        for x, y in zip(evol_x_1, evol_y_1):
            ax.plot(x, y, color=color, marker=(marker_size, 2), markersize=12)
    else:
        for x in evol_x_1:
            ax.plot(x, evol_y_1, color=color, marker=(marker_size, 2), markersize=12)
    pyplot.show()
    fig, ax = pyplot.subplots()
    color = 'red'
    evol_x_2, evol_y_2 = evol_results_2
    if isinstance(evol_y_2, list):
        ax.plot(evol_x_2, evol_y_2, color=color, marker=(marker_size, 2), markersize=12)
    else:
        for x in evol_x_2:
            ax.plot(x, evol_y_2, color=color, marker=(marker_size, 2), markersize=12)

    ax.set_xlabel(f'approximated qubo entries in percent')
    ax.set_title(title)
    #pyplot.show()


def plot_average_fitness(avg_fitness_list):
    fig, ax = pyplot.subplots()
    ax.plot([i + 1 for i in range(len(avg_fitness_list))], avg_fitness_list, color="blue", markersize=8)
    ax.set_ylabel('Average Fitness')
    ax.set_xlabel(f'Generation')
    ax.set_title(f'Average Fitness', fontsize=12)
    pyplot.show()


# Plot the original QUBO together with the derived mask
def compare_qubo_to_mask(qubo: np.array, qubo_mask: np.array, title: str):
    fig, axes = pyplot.subplots(ncols=2)
    qubo_list = [qubo, qubo_mask]
    #for i in range(2):
    work_qubo = qubo_list[0]
    work_ax = axes[0]
    mesh = work_ax.pcolormesh(work_qubo, cmap="plasma")
    work_ax.set_xticks([i for i in range(work_qubo.shape[0])], minor=True)
    work_ax.set_yticks([i for i in range(work_qubo.shape[0])], minor=True)
    work_ax.grid(True, color="black", which="both", linewidth=0.1, linestyle="-")
    work_ax.set_title(f'QUBO heatmap for {title}')
    fig.colorbar(mesh, ax=work_ax)
    work_ax.invert_yaxis()

    work_qubo = qubo_list[1]
    work_ax = axes[1]
    mesh = work_ax.pcolormesh(work_qubo, cmap="plasma")
    work_ax.set_xticks([i for i in range(work_qubo.shape[0])], minor=True)
    work_ax.set_yticks([i for i in range(work_qubo.shape[0])], minor=True)
    work_ax.grid(True, color="black", which="both", linewidth=0.1, linestyle="-")
    work_ax.set_title(f'QUBO approximation mask for {title}')
    fig.colorbar(mesh, ax=work_ax)
    work_ax.invert_yaxis()
    pyplot.show()


# Heatmap to visualize a QUBO
def qubo_heatmap(qubo_matrix: np.array, title="QUBO Heatmap"):
    fig, ax = pyplot.subplots()
    mesh = ax.pcolormesh(qubo_matrix, cmap="plasma")
    ax.set_xticks([i for i in range(qubo_matrix.shape[0])], minor=True)
    ax.set_yticks([i for i in range(qubo_matrix.shape[0])], minor=True)
    ax.grid(True, color="black", which="both", linewidth=0.1, linestyle="-")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax)
    pyplot.gca().invert_yaxis()
    pyplot.show()


# Methods to plot and validate the results of a TSP problem instance
def get_best_solution(solutions: list[list[int]], dist_matrix: list[list[float]]):
    n = len(dist_matrix)
    paths = [[0 for i in range(n)] for sol in solutions]
    min_length = np.inf
    best_path = []
    for h in range(len(solutions)):
        for i in range(n):
            for j in range(n):
                if solutions[h][i * n + j]:
                    paths[h][j] = i

    for path in paths:
        length = 0
        for i in range(n-1):
            length += dist_matrix[path[i]][path[i + 1]]
        length += dist_matrix[path[0]][path[n - 1]]
        if length < min_length:
            min_length = length
            best_path = path
    #print(f'The solutions suggests a path length of {min_length:.4f}!')
    return round(min_length, 4), best_path


def plot_cities_with_solution(cities: list[list[float, float]], path: list[int]):
    #print(solution)
    fig, ax = pyplot.subplots()
    for idx, city in enumerate(cities):
        ax.plot(city[0], city[1], marker='x')
        pyplot.text(city[0], city[1], str(idx + 1))

    n = len(cities)

    for i in range(n-1):
        ax.plot([cities[path[i]][0], cities[path[i + 1]][0]],
                [cities[path[i]][1], cities[path[i + 1]][1]])
    ax.plot([cities[path[n - 1]][0], cities[path[0]][0]],
            [cities[path[n - 1]][1], cities[path[0]][1]])

    #pyplot.show()


def bruteforce_verification(cities: list[list[float, float]], distance_matrix: list[list[float]]):
    # print(f'\n\nStarting brute-force verification')
    # all permutations of city choices
    permutations = np.array(list(itertools.permutations(range(len(cities)), len(cities))))

    #set up empty total distance vector
    all_perm_dists = [0]*permutations.shape[0]

    # calculate all distances
    for i in range(permutations.shape[0]):
        for j in range(permutations.shape[1]-1):
            c1 = permutations[i, j]   # choice 1
            c2 = permutations[i, j+1]  # choice 2
            all_perm_dists[i] += distance_matrix[c1][c2]
        all_perm_dists[i] += distance_matrix[permutations[i, 0]][permutations[i, len(cities) - 1]]

    #distances are rounded to 4 decimals because of floating point issues
    min_dist = round(min(all_perm_dists), 4)
    min_routes = permutations[np.argwhere(np.around(all_perm_dists, 4) == min_dist).flatten()]
    min_routes_alph = [[min_routes[i, j] for j in range(min_routes.shape[1])]
                       for i in range(min_routes.shape[0])]

    # Note that for the brute force analyses, the final return back to the initial city is included in the displayed output.
    #print(f'The shortest cycle is {min_dist:.4f} units')  # \nFor routes:\n{min_routes_alph}')
    return min_dist, min_routes_alph[0]
