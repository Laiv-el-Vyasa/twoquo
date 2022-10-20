import numpy as np
from config import load_cfg
from pipeline_util import QUBOGenerator
from recommendation import RecommendationEngine

cfg = load_cfg(cfg_id='test_evol')
qubo_size = cfg['pipeline']['problems']['qubo_size']
engine = RecommendationEngine(cfg=cfg)
generator = QUBOGenerator(cfg)

solver = 'qbsolv_simulated_annealing'


def get_linearized_qubos():
    qubos = get_problem_qubos()
    qubo_energy_list = []
    for qubo in qubos:
        _, min_energy = solve_qubo(qubo)
        qubo_energy_list.append((linearize_qubo(qubo), qubo, min_energy))
    return qubo_energy_list


def get_quality_of_approxed_qubo(linearized_approx, qubo, min_energy):
    approxed_qubo = apply_approximation_to_qubo(linearized_approx, qubo)
    solutions = solve_qubo(approxed_qubo)
    return get_min_solution_quality(solutions, qubo, min_energy)


def linearize_qubo(qubo):
    linearized_qubo = []
    for i in range(qubo_size):
        for j in range(i):
            linearized_qubo.append(qubo[i][j])
    return linearized_qubo


def apply_approximation_to_qubo(linearized_approx, qubo):
    approxed_qubo = np.zeros((qubo_size, qubo_size))
    linear_index = 0
    for i in range(qubo_size):
        for j in range(i):
            if linearized_approx[linear_index] > 0:
                approxed_qubo[i][j] = qubo[i][j]
                approxed_qubo[j][i] = qubo[j][i]
            linear_index += 1
    return approxed_qubo


def get_problem_qubos():
    qubos, labels, problems = generator.generate()
    return qubos


def solve_qubo(qubo):
    metadata = engine.recommend(qubo)
    print(metadata.solutions[solver], metadata.energies[solver])
    return metadata.solutions[solver], np.min(metadata.energies[solver])


def get_min_solution_quality(solutions, qubo, min_energy):
    solution_quality_array = []
    for solution in solutions[0]:
        solution_quality_array.append(get_solution_quality(solution.dot(qubo.dot(solution)),min_energy))
    return np.min(solution_quality_array)


def get_solution_quality(energy, min_energy):
    return (min_energy - energy) / min_energy