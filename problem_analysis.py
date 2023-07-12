import numpy as np
from matplotlib import pyplot

from config import load_cfg
#from evolution.evolution_util import check_solution
from pipeline_util import QUBOGenerator
from recommendation import RecommendationEngine
from visualisation import qubo_heatmap

import torch, os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
cfg = load_cfg(cfg_id='test_evol_sgi_small')
qubo_size = cfg['pipeline']['problems']['qubo_size']
engine = RecommendationEngine(cfg=cfg)
generator = QUBOGenerator(cfg)
problem_name = cfg['pipeline']['problems']['problems'][0]
solver = 'qbsolv_simulated_annealing'

qubos, labels, problems = generator.generate()
print(qubos[0])
print(problems[0])
np.set_printoptions(threshold=np.inf)


if problem_name == 'NPP':
    number_list = []
    length_list = []
    max_number = 0
    for problem, qubo in zip(problems, qubos):
        print(problem)
        qubo_heatmap(qubo, name='QUBO heatmap for NPP')
        qubo_heatmap(np.triu(qubo), name='QUBO heatmap for NPP')
        length_list.append(len(problem['numbers']))
        number_list.extend(problem['numbers'])
        if np.max(problem['numbers']) > max_number:
            max_number = np.max(problem['numbers'])
    print(max_number)
    print(length_list)
    #pyplot.hist(number_list, bins=max_number)
    #pyplot.xlabel('Numbers in NP-problems')
    #pyplot.ylabel('Count')
    #pyplot.title(f'Number count in {len(problems)} NP-problems ({qubo_size}x{qubo_size})')
    #pyplot.show()
elif problem_name == 'MC':
    edge_number_list = []
    max_number = 0
    min_number = np.inf
    for problem, qubo in zip(problems, qubos):
        graph = problem['graph']
        edge_number = len(list(graph.edges))
        edge_number_list.append(edge_number)
        print(graph.nodes)
        print(graph.edges)
        qubo_heatmap(qubo, name='QUBO heatmap for MC')
        qubo_heatmap(np.triu(qubo), name='QUBO heatmap for MC')
        if edge_number > max_number:
            max_number = edge_number
        if edge_number < min_number:
            min_number = edge_number

    pyplot.hist(edge_number_list, bins=max_number - min_number)
    pyplot.xlabel('Edge number in MC-problems')
    pyplot.ylabel('Count')
    pyplot.title(f'Edge number count in {len(problems)} MC-problems ({qubo_size}x{qubo_size})')
    pyplot.show()
elif problem_name == 'TSP':
    for problem, qubo in zip(problems, qubos):
        print(problem)
        qubo_heatmap(qubo, name='QUBO heatmap for TSP')
        qubo_heatmap(np.triu(qubo), name='QUBO heatmap for TSP')
elif problem_name == 'GC':
    for problem, qubo in zip(problems, qubos):
        print(problem['graph'].nodes)
        print(problem['graph'].edges)
        qubo_heatmap(qubo, name='QUBO heatmap for GC')
        qubo_heatmap(np.triu(qubo), name='QUBO heatmap for GC')
elif problem_name == 'M3SAT':
    for problem, qubo in zip(problems, qubos):
        print(problem)
        qubo_heatmap(qubo, name='QUBO heatmap for M3SAT')
        qubo_heatmap(np.triu(qubo), name='QUBO heatmap for M3SAT')
elif problem_name == 'EC':
    for problem, qubo in zip(problems, qubos):
        print(problem)
        qubo_heatmap(qubo, name='QUBO heatmap for EC')
        qubo_heatmap(np.triu(qubo), name='QUBO heatmap for EC')
elif problem_name == 'SGI':
    for problem, qubo in zip(problems, qubos):
        print(problem['graph1'].nodes)
        print(problem['graph1'].edges)
        print(problem['graph2'].nodes)
        print(problem['graph2'].edges)
        qubo_heatmap(qubo, name='QUBO heatmap for SGI')
        qubo_heatmap(np.triu(qubo), name='QUBO heatmap for SGI')
else:
    print(problems[0])
    print(qubos[0])
    metadata = engine.recommend(qubos[0])
    print(metadata.solutions)
    #print('Solution value: ', check_solution(metadata.solutions[solver][0], metadata.solutions[solver][0], problems[0]))
    qubo_heatmap(qubos[0])