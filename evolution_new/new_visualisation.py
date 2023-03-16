from matplotlib import pyplot
import matplotlib.lines as lines
import numpy as np
from config import load_cfg


def visualize_evol_results(baseline_approx_data, percent_list, evol_results, model_name,
                           problem, size, solver, steps, boxplot=False):
    fig, ax = pyplot.subplots()
    legend_lines = []
    color = 'black'
    ax.plot(percent_list, baseline_approx_data, color=color, markersize=8)
    legend_lines.append(lines.Line2D([], [], color=color, markersize=8, label=f'step-wise approximation, {steps} steps'))
    evol_x, evol_y = evol_results
    if not boxplot:
        evol_x = np.mean(evol_x)
    marker_size = 4

    if not boxplot:
        ax.plot(evol_x, evol_y, color=color, marker=(marker_size, 2), markersize=12)
    else:
        for x in evol_x:
            ax.plot(x, evol_y, color=color, marker=(marker_size, 2), markersize=12)
        #pyplot.boxplot(evol_x, positions=np.mean(evol_x), vert=False)
    legend_lines.append(lines.Line2D([], [], color=color, linestyle='None',
                                     markersize=12, marker=(marker_size, 2), label=f'{model_name}'))
    ax.legend(handles=legend_lines)
    ax.set_ylabel(f'best energy found in percent')
    ax.set_xlabel(f'approximated qubo entries in percent')
    ax.set_title(f'Approximation quality, Learned Models, Problem: {problem}, '
                 f'Size: {size}, Solver: {solver}', fontsize=12)
    pyplot.show()


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


def plot_points(point_array: list[list[float, float]]):
    fig, ax = pyplot.subplots()
    for point in point_array:
        ax.plot(point[0], point[1], marker='x')
    pyplot.show()
