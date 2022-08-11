"""Tabu Search solver shipped with TooQuO. Uses the `dwave-tabu` package.
"""
import tabu
from dwave_qbsolv import QBSolv


def solve(Q, *args, **kwargs):
    """qbsolv tabu search
    """
    sampler = tabu.TabuSampler()
    resp = QBSolv().sample_qubo(Q, **kwargs, solver=sampler).record
    print(resp['sample'])
    return resp['sample'], resp['energy']
