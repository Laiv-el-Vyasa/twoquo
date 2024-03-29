import numpy as np
from matplotlib import pyplot

from config import load_cfg
#from evolution.evolution_util import check_solution
from pipeline_util import QUBOGenerator
from recommendation import RecommendationEngine
from visualisation import qubo_heatmap

import torch, os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cfg = load_cfg(cfg_id='test_evol_ec_large')
qubo_size = cfg['pipeline']['problems']['qubo_size']
engine = RecommendationEngine(cfg=cfg)
generator = QUBOGenerator(cfg)
problem_name = cfg['pipeline']['problems']['problems'][0]
solver = 'qbsolv_simulated_annealing'

qubos, labels, problems = generator.generate()
print(qubos)
#print(problems[0])
np.set_printoptions(threshold=np.inf)


if problem_name == 'NP':
    number_list = []
    max_number = 0
    for problem in problems:
        number_list.extend(problem['numbers'])
        if np.max(problem['numbers']) > max_number:
            max_number = np.max(problem['numbers'])
    print(max_number)
    pyplot.hist(number_list, bins=max_number)
    pyplot.xlabel('Numbers in NP-problems')
    pyplot.ylabel('Count')
    pyplot.title(f'Number count in {len(problems)} NP-problems ({qubo_size}x{qubo_size})')
    pyplot.show()
elif problem_name == 'MC':
    edge_number_list = []
    max_number = 0
    min_number = np.inf
    for problem in problems:
        graph = problem['graph']
        edge_number = len(list(graph.edges))
        edge_number_list.append(edge_number)
        if edge_number > max_number:
            max_number = edge_number
        if edge_number < min_number:
            min_number = edge_number

    pyplot.hist(edge_number_list, bins=max_number - min_number)
    pyplot.xlabel('Edge number in MC-problems')
    pyplot.ylabel('Count')
    pyplot.title(f'Edge number count in {len(problems)} MC-problems ({qubo_size}x{qubo_size})')
    pyplot.show()
else:
    print(problems[0])
    print(qubos[0])
    metadata = engine.recommend(qubos[0])
    print(metadata.solutions)
    #print('Solution value: ', check_solution(metadata.solutions[solver][0], metadata.solutions[solver][0], problems[0]))
    qubo_heatmap(qubos[0])