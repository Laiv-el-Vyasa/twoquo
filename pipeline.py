import os
import collections
import random

from typing import List

from config import load_cfg
from approximation import get_approximated_qubos
from pipeline_util import QUBOGenerator
from recommendation import RecommendationEngine
import numpy as np
from visualisation import qubo_heatmap, approx_quality_graphs
from operator import add


def get_problem_score(problem_name, problem, solution=None):
    if problem_name == 'NP':
        score = get_np_problem_score(problem, solution)
    elif problem_name == 'MC':
        score = get_mc_problem_score(problem, solution)
    elif problem_name == "M2SAT" or problem_name == "M3SAT":
        score = get_sat_problem_score(problem, solution)
    else:
        if solution is None:
            score = 1
        else:
            score = 0
    return score


def get_np_problem_score(problem, solution):
    number_list = problem['numbers']
    if solution is None:
        score = sum(number_list)
    else:
        score_1 = sum([number * sol_value for number, sol_value in zip(number_list, solution)])
        score_2 = sum([number * sol_value for number, sol_value in zip(number_list, map(lambda x: (x + 1) % 2, solution))])
        score = np.absolute(score_1 - score_2)
    return score


def get_mc_problem_score(problem, solution):
    graph = problem['graph']
    score = 0
    if solution is None:
        return 0
    else:
        nodes = list(graph.nodes)
        edges = list(graph.edges)
        set_1 = []
        set_2 = []
        for index, i in enumerate(solution):
            if i:
                set_1.append(nodes[index])
            else:
                set_2.append(nodes[index])
        #print(solution)
        #print(set_1, set_2)
        for node_1, node_2 in edges:
            if (node_1 in set_1 and node_2 in set_2) or (node_1 in set_2 and node_2 in set_1):
                score +=1
        #print('Score: ' + str(score))
        return score


def get_sat_problem_score(problem, solution):
    clauses = problem["clauses"]
    true_clauses = len(clauses)
    if solution is not None:
        for clause in clauses:
            assigned_sum = 0
            for variable_index, affirmed in clause:
                assigned_value = solution[variable_index]
                if affirmed:
                    assigned_sum += assigned_value
                else:
                    assigned_sum += ((assigned_value + 1) % 2)
            #print("Assigned sum: " + str(assigned_sum))
            if assigned_sum == 0:
                true_clauses-= 1
    else:
        return 0
    return true_clauses


def get_best_score(problem_name, problem, solutions):
    score_list = [get_problem_score(problem_name, problem, sol) for sol in solutions[0]]
    return np.max(score_list)


def get_worst_metadata(qubo, eng):
    inverse_qubo = qubo * (-1)
    return eng.recommend(inverse_qubo)


def get_worst_energy(qubo, engine):
    metadata = get_worst_metadata(qubo, engine)
    worst_energy = np.inf * (-1)
    for solver_solutions in metadata.solutions.values():
        for solution in solver_solutions[0]:
            energy = solution.dot(qubo.dot(solution))
            if energy > worst_energy:
                worst_energy = energy
    return worst_energy


def get_max_solution_quality(solutions, qubo, best_energy, worst_energy):
    solution_quality_array = []
    for solution in solutions[0]:
        solution_quality_array.append(get_solution_quality(solution.dot(qubo.dot(solution)), best_energy, worst_energy))
    return np.max(solution_quality_array)


def get_solution_quality(energy, best_energy, worst_energy):
    return 1 - ((best_energy - energy) / (best_energy - worst_energy))


if __name__ == '__main__':
    approximation_steps = 200
    approx_fixed_number = True
    print_bool = False
    save_bool = True
    cfg = load_cfg(cfg_id='test')
    gen = QUBOGenerator(cfg)

    qubos, labels, problems = gen.generate()
    qubo_problem_list = list(zip(qubos, labels, problems))
    random.shuffle(qubo_problem_list)
    problem_names = cfg['pipeline']['problems']['problems']
    #solver_name = 'qbsolv_simulated_annealing'

    eng = RecommendationEngine()
    percentage_steps = [(x + 1) / (approximation_steps + 1) for x in range(approximation_steps)]
    overall_data = {'approximation_steps': percentage_steps}

    for count, (qubo, label, problem) in enumerate(qubo_problem_list):
        print(f"Calculating problem {count + 1}:")
        problem_name = problem_names[label]
        if not problem_name in overall_data.keys():
            overall_data[problem_name] = []
        metadata = eng.recommend(qubo)
        metadata.problem = label
        metadata.approx = approximation_steps
        metadata.approx_strategy = approx_fixed_number

        worst_energy = get_worst_energy(qubo, eng)
        best_energy_solver_dict = {}
        for solver in metadata.energies:
            best_energy_solver_dict[solver] = np.min(metadata.energies[solver])
        metadata.solution_quality = {solv: 1 for solv in metadata.solutions}



        solution_score = get_best_score(problem_name, problem, metadata.solutions[list(metadata.solutions.keys())[0]])
        worst_score = get_problem_score(problem_name, problem)
        specific_problem = {
            'qubo': qubo,
            'solution': metadata.solutions[list(metadata.solutions.keys())[0]][0],
            'solution_energy': metadata.energies[list(metadata.solutions.keys())[0]][0],
            'best_score': solution_score,
            'worst_score': worst_score,
            'approximation': []
        }

        approx_qubos = get_approximated_qubos(qubo, approx_fixed_number, percentage_steps)
        percentage_approxed = 0
        for approx_step in approx_qubos:
            approx_data = approx_qubos[approx_step]

            approx_qubo = approx_data['qubo']
            approx_metadata = eng.recommend(approx_qubo)

            #metadata.approx = approx_step
            #metadata.approx_strategy = approx_fixed_number
            solution_quality_dict = {}
            for solver in approx_metadata.solutions:
                solution_quality_dict[solver] = \
                    get_max_solution_quality(approx_metadata.solutions[solver],
                                             qubo, best_energy_solver_dict[solver], worst_energy)
            metadata.approx_solution_quality[approx_step] = solution_quality_dict
            #if save_bool:
            #    eng.save_metadata(metadata)

            reverse_energy_appr = get_max_solution_quality(approx_metadata.solutions[list(approx_metadata.solutions.keys())[0]],
                                                           qubo, best_energy_solver_dict[list(best_energy_solver_dict.keys())[0]], worst_energy)
            approx_solution_score = get_best_score(problem_name, problem, approx_metadata.solutions[list(approx_metadata.solutions.keys())[0]])
            if print_bool:
                percentage_approxed += (approx_data["approximations"] / approx_data['size'])
                print(str(approx_step) + '. Approx: ' + str(approx_data["approximations"]) +
                      ' Approximations, ' + str(percentage_approxed) + ' of total')
                print(str(approx_step) + '. Approx: Approx solution: ' +
                      str(approx_metadata.solutions[list(approx_metadata.solutions.keys())[0]][0][0]))
                print(str(approx_step) + '. Approx: Reverse Energy approx: ' +
                      str(approx_metadata.energies[list(approx_metadata.energies.keys())[0]][0][0]))
                print(str(approx_step) + '. Approx: Approximation quality: ' + str(reverse_energy_appr))

                print(str(approx_step) + '. Approx: Approximation score: ' + str(approx_solution_score))
                print(str(approx_step) + '. Approx: Approximation score quality: ' +
                  str(1 - ((approx_solution_score - solution_score) / (worst_score - solution_score))))

            approx_data['solution'] = approx_metadata.solutions[list(approx_metadata.solutions.keys())[0]][0]
            approx_data['rel_energy'] = reverse_energy_appr
            approx_data['rel_score'] = 1 - ((approx_solution_score - solution_score) / (worst_score - solution_score))

            specific_problem['approximation'].append(approx_data)

        if save_bool:
            eng.save_metadata(metadata)

        overall_data[problem_name].append(specific_problem)
        overall_data['solver'] = list(metadata.solutions.keys())[0]


    #print(overall_data)
    approx_quality_graphs(overall_data, approx_fixed_number)


def save_metadata():
    pass


def add_to_display_data():
    pass


