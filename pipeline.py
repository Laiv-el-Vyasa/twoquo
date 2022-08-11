import os
import collections

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


def get_worst_metadata(qubo, eng):
    inverse_qubo = qubo * (-1)
    metadata, solution, energy = eng.recommend(inverse_qubo)
    return metadata


def get_worst_energy_from_solutions(worst_solutions, qubo):
    worst_energy = np.inf * (-1)
    for solver_solutions in worst_solutions.values():
        for solution in solver_solutions:
            energy = solution.dot(qubo.dot(solution))
            if energy > worst_energy:
                worst_energy = energy
    return worst_energy


def get_average_solution_quality(solutions, qubo, best_energy, worst_energy):
    average_solution_energy = np.average(solutions.dot(qubo.dot(solutions)))
    return 1 - ((best_energy - average_solution_energy) / (worst_energy - best_energy))


if __name__ == '__main__':
    approximation_steps = 10
    approx_fixed_number = False
    cfg = load_cfg(cfg_id='test')
    gen = QUBOGenerator(cfg)

    qubos, labels, problems = gen.generate()
    problem_names = cfg['pipeline']['problems']['problems']

    eng = RecommendationEngine()
    percentage_steps = [(x + 1) / (approximation_steps + 1) for x in range(approximation_steps)]
    overall_data = {'approximation_steps': percentage_steps}

    for qubo, label, problem in zip(qubos, labels, problems):
        problem_name = problem_names[label]
        #print('Problem: ' + str(problem))
        if not problem_name in overall_data.keys():
            overall_data[problem_name] = []
        metadata, solutions, energy = eng.recommend(qubo)
        metadata.Q_initial = qubo
        metadata.approx = 0
        metadata.approx_strategy = approx_fixed_number
        metadata.approx_percent = 0

        worst_metadata = get_worst_metadata(qubo, eng)
        worst_energy = get_worst_energy_from_solutions(worst_metadata.solutions, qubo)
        best_energy_solver_dict = {}
        for solver in metadata.energies:
            best_energy_solver_dict[solver] = np.mean(metadata.energies[solver])
        metadata.average_solution_quality = {solv: 1 for solv in metadata.solutions}

        eng.save_metadata(metadata)

        one_solution = solutions[0]
        print('Solution: ' + str(one_solution))
        reverse_energy_sol = one_solution.dot(qubo.dot(one_solution))
        print('Reverse Energy: ' + str(reverse_energy_sol))

        solution_score = get_problem_score(problem_name, problem, one_solution)
        #print('Solution score: ' + str(solution_score))
        worst_score = get_problem_score(problem_name, problem)
        #print('Worst score: ' + str(worst_score))
        specific_problem = {
            'qubo': qubo,
            'solution': one_solution,
            'solution_energy': reverse_energy_sol,
            'best_score': solution_score,
            'worst_score': worst_score,
            'approximation': []
        }

        #print(qubo)
        #qubo_heatmap(qubo)
        approx_qubos = get_approximated_qubos(qubo, approx_fixed_number, percentage_steps)
        percentage_approxed = 0
        for approx_step in approx_qubos:
            approx_data = approx_qubos[approx_step]

            approx_qubo = approx_data['qubo']
            #print(approx_qubo)
            #qubo_heatmap(approx_qubo)

            metadata, soluts, energy = eng.recommend(approx_qubo)

            metadata.Q_initial = qubo
            metadata.approx = approx_step
            eng.save_metadata(metadata)

            reverse_energy_appr = soluts[0].dot(qubo.dot(soluts[0]))
            approx_solution_score = get_problem_score(problem_name, problem, soluts[0])
            percentage_approxed += (approx_data["approximations"] / approx_data['size'])
            print(str(approx_step) + '. Approx: ' + str(approx_data["approximations"]) +
                  ' Approximations, ' + str(percentage_approxed) + ' of total')
            print(str(approx_step) + '. Approx: Approx solution: ' + str(soluts[0]))
            print(str(approx_step) + '. Approx: Reverse Energy approx: ' + str(reverse_energy_appr))
            print(str(approx_step) + '. Approx: Approximation quality: ' +
                  str(1 - ((reverse_energy_appr - reverse_energy_sol) / (worst_energy - reverse_energy_sol))))
            #energy_quality_list.append(reverse_energy_appr / reverse_energy_sol)

            print(str(approx_step) + '. Approx: Approximation score: ' + str(approx_solution_score))
            print(str(approx_step) + '. Approx: Approximation score quality: ' +
                  str(1 - ((approx_solution_score - solution_score) / (worst_score - solution_score))))
            #score_quality_list.append(1 - ((approx_solution_score - solution_score) / (worst_score - solution_score)))

            approx_data['solution'] = soluts[0]
            approx_data['rel_energy'] = 1 - ((reverse_energy_appr - reverse_energy_sol) / (worst_energy - reverse_energy_sol))
            approx_data['rel_score'] = 1 - ((approx_solution_score - solution_score) / (worst_score - solution_score))

            specific_problem['approximation'].append(approx_data)

        overall_data[problem_name].append(specific_problem)


    #print(overall_data)
    approx_quality_graphs(overall_data, approx_fixed_number)

