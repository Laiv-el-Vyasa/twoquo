"""Simulated annealing solver shipped with TooQuO. Uses the `dwave-neal`
package.
"""
import dimod
from dwave_qbsolv import QBSolv


def solve(Q, *args, **kwargs):
    """qbsolv simulated annealing
    """
    sampler = dimod.ExactSolver()
    resp = QBSolv().sample_qubo(Q, **kwargs, solver=sampler).record
    return resp['sample'], resp['energy']
