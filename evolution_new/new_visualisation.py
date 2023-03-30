import matplotlib.figure
from matplotlib import pyplot
import matplotlib.lines as lines
import numpy as np
from config import load_cfg
import itertools


def visualisation_pipeline(evaluation_dict: dict):
    fig, ax = pyplot.subplots()
    legend_lines = []
    if 'baseline_data' in evaluation_dict:
        ax, legend_lines = add_baseline(ax, evaluation_dict['baseline_data'], legend_lines)

    ax, legend_lines = plot_points(ax, evaluation_dict['evaluation_results'], legend_lines)
    ax.legend(handles=legend_lines)
    ax.set_ylabel(evaluation_dict['ylabel'])
    ax.set_xlabel(evaluation_dict['xlabel'])
    ax.set_title(evaluation_dict['title'], fontsize=12)
    pyplot.show()


def add_baseline(ax: matplotlib.axes.Axes, baseline_data: dict, legend_lines: list) -> tuple:
    color = 'black'
    ax.plot(baseline_data['percent_list'], baseline_data['baseline_approx_data'], color=color, markersize=8)
    legend_lines.append(lines.Line2D([], [], color=color, markersize=8, label=baseline_data['legend']))
    return ax, legend_lines


def plot_points(ax: matplotlib.axes.Axes, evaluation_results: dict, legend_lines: list) -> tuple:
    for evol_result in evaluation_results:
        color = evol_result['color']
        marker = evol_result['marker']
        for x, y in zip(evol_result['evol_x'], evol_result['evol_y']):
            ax.plot(x, y, color=color, marker=(marker, 2), markersize=12)
            legend_lines.append(lines.Line2D([], [], color=color, linestyle='None',
                                             markersize=12, marker=(marker, 2), label=evol_result['label']))
    return ax, legend_lines


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
