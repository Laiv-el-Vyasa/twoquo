"""
The purpose of this file is to export a global dictionary that can be used
and modified in other modules such as the recommendation module.

Further, this is part of a plugin system that allows configuring solvers
dynamically.

This is based on a plugin system that is configured in config.json:

    "solvers": {
        "simulated_annealing": {
            "mod": "tooquo.solver.simulated_annealing",
            "type": "classical",
            "max_vars": 0,
            "max_connectivity": 0,
            "kwargs": {
                "num_repeats": 100
            },
            "repeats": 10
        },
        "tabu_search": {
            "mod": "tooquo.solver.tabu_search",
            "type": "classical",
            "max_vars": 0,
            "max_connectivity": 0,
            "kwargs": {},
            "repeats": 10
        },
        "brute_force": {
            "mod": "tooquo.solver.brute_force",
            "type": "classical",
            "max_vars": 0,
            "max_connectivity": 0,
            "kwargs": {
                "solver_limit": 16
            },
            "repeats": 1
        }
    }

It is easy to include new solvers by following two steps:

1. Create a new python file that includes the actual solver, e.g.:

        import tabu
        from dwave_qbsolv import QBSolv

        def solve(Q, *args, **kwargs):
            sampler = tabu.TabuSampler()
            resp = QBSolv().sample_qubo(Q, **kwargs, solver=sampler).record
            return resp['sample'], resp['energy']

    The return value of the solve function should be a list of solution
    vectors (e.g. when sampling from QC HW) and a list of corresponding
    energies.
    If there is just one solution vector (e.g. when bruteforcing), the
    return value should still be a list, but simply with just one entry.

2. Create an entry in config.json.
"""
import copy
import importlib


class Solver:
    def __init__(self, name, solve_func, solver_cfg):
        """Create a Solver instance.

        Args:
            name: The unique identifier of this Solver, e.g. `bruteforce` or
                `simulated_annealing`.
            solve_func: The solution function found in the python file of the
                solver. It should accept a QUBO and return a list of solutions
                and a list of corresponding energies. Example:

                    def solve(Q, *args, **kwargs):
                        sampler = tabu.TabuSampler()
                        resp = QBSolv().sample_qubo(
                            Q, **kwargs, solver=sampler
                        ).record
                        return resp['sample'], resp['energy']

            solver_cfg: The additional configuration of the solver.
                Should at least include the following keys:

                    {
                        "max_vars": 0,
                        "max_connectivity": 0,
                        "repeats": 0
                    }

                `max_vars` is the maximum number of variables supported by this
                solver. If set to 0, an infinite number of variables are
                supported. `max_connectivity` is the maximum connectivity
                supported by the solver, with 0 also denoting infinity.
                `repeats` is the number of times the solver should be called
                for a given QUBO.


        """
        self.name = name
        self.solve_func = solve_func
        self.solver_cfg = solver_cfg

        self.enabled = solver_cfg['enabled']
        self.max_vars = solver_cfg['max_vars']
        self.type_ = solver_cfg['type']
        self.max_connectivity = solver_cfg['max_connectivity']
        self.kwargs = solver_cfg['kwargs']
        qpu_solver = solver_cfg.get('qpu_solver', '')
        if qpu_solver:
            qpu_solver_parts = qpu_solver.split('.')
            mod = importlib.import_module('.'.join(qpu_solver_parts[:-1]))
            self.kwargs['qpu_solver'] = mod.__dict__[qpu_solver_parts[-1]]
        self.repeats = solver_cfg['repeats']

    def __call__(self, Q):
        return self.solve_func(Q, **self.kwargs)


"""
A dictionary of solvers, with key being a unique identifier and the value
being a function that can be called. The solve function shall accept a QUBO and
optionally **kwargs and return the solution vector of the QUBO.
"""
SOLVERS = {}


def load_solvers(cfg):
    """Load the solvers from the configuration dictionary.

    Args:
        cfg: The parsed JSON configuration as a dict.

    This function modifies the global SOLVERS dictionary.

    This is based on a plugin system that is configured in config.json.
    To find out more, check out the documentation of the solver subpackage
    (``tooquo.solver.solver``).

    This parses the configuration entry for a solver and creates a ``Solver``
    instance. This instance has a solver name, a solver function, which is
    loaded from the location pointed at by the `mod` key, and metadata.

    The SOLVERS dictionary is simply a mapping from solver names to their
    Solver instances. The Solver instance can be used to actually solve a
    QUBO by calling it:

        solutions, energies = SOLVERS['bruteforce'](qubo)

    """
    cfg = copy.deepcopy(cfg)
    for solver_name, solver_cfg in cfg.get('solvers', {}).items():
        mod = importlib.import_module(solver_cfg.pop("mod"))
        SOLVERS[solver_name] = Solver(solver_name, mod.solve, solver_cfg)
