"""Simulated annealing solver shipped with TooQuO. Uses the `dwave-neal`
package.
"""
import neal
from dwave_qbsolv import QBSolv


def solve(Q, *args, **kwargs):
    """qbsolv simulated annealing
    """
    sampler = neal.SimulatedAnnealingSampler()
    resp = QBSolv().sample_qubo(Q, **kwargs, solver=sampler).record
    return resp['sample'], resp['energy']
