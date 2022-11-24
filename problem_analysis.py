import numpy as np
from matplotlib import pyplot

from config import load_cfg
from pipeline_util import QUBOGenerator
from recommendation import RecommendationEngine

import torch, os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cfg = load_cfg(cfg_id='test_evol_mc')
qubo_size = cfg['pipeline']['problems']['qubo_size']
engine = RecommendationEngine(cfg=cfg)
generator = QUBOGenerator(cfg)
problem_name = cfg['pipeline']['problems']['problems'][0]

qubos, labels, problems = generator.generate()


if problem_name == 'NP':
    number_list = []
    max_number = 0
    for problem in problems:
        number_list.extend(problem['numbers'])
        if np.max(problem['numbers']) > max_number:
            max_number = np.max(problem['numbers'])


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