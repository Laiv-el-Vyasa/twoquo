"""Intercept QUBOs, analyze and modify them, and execute configured solvers.
"""
import time
import traceback
import multiprocessing

import numpy as np

from solver.solver import SOLVERS
#from tooquo.loggin import GLOBAL_LOGGER


def compute_qubo_multiproc(
        solver,
        solver_name,
        Q,
        sync_dict,
        wrong_energy_check):
    """Used by ``multiprocessing`` to solve QUBOs in a parallel fashion.

    Args:
        solver: The Solver instance.
        solver_name: The name of the solver.
        Q: The QUBO.
        sync_dict: An instance of ``multiprocessing.Manager.dict`` which is
            modified inplace and propagated back in a synchronous way by the
            multiprocessing module such that the solver results can be
            collected by the parent process.
        wrong_energy_check: Whether to verify the energy that was returned by
            the solver.
    """
    start_time = time.time()
    try:
        sol, energy_ = solver(Q)
    except:
        traceback.print_exc()
        return
    total_time = time.time() - start_time
    energy = [sol[0].transpose() @ Q @ sol[0]]
    if wrong_energy_check and energy_[0] != energy[0]:
        raise Exception(
            "Solver " + str(solver_name) + " returned wrong energy: " +
            energy_[0] + " != " + energy[0]
        )
    sync_dict[solver_name]["runtimes"].append(total_time)
    sync_dict[solver_name]["solutions"].append(sol)
    sync_dict[solver_name]["energies"].append(energy)


class Interceptor:
    def __init__(self, global_cfg, solvers, logger=None):
        """Create an Interceptor instance.

        Args:
            solvers: A list of solver keys, with a single key being a unique
                identifier in the global solvers dictionary.
        """
        self.global_cfg = global_cfg
        if isinstance(solvers, dict):
            solvers = list(solvers.keys())
        self.solvers = solvers
        #if logger is not None:
            #self.logger = logger
        #else:
            #self.logger = GLOBAL_LOGGER.configure(self.global_cfg)

        self.cfg = global_cfg['interceptor']
        self.solver_performance_metric = self.cfg['solver_performance_metric']
        self.use_multi_proc = self.cfg['use_multi_proc']
        self.wrong_energy_check = self.cfg['wrong_energy_check']

    def is_adquate_solver(self, solver, metadata):
        """Checks whether a given solver is adequate for a given QUBO.

        For instance, if the QUBO has a larger maximum connectivity than the
        maximum one supported by the solver, the solver is deemed to be
        inadequate.

        Returns a boolean indicating whether the solver is adequate for solving
        the given QUBO.
        """
        # TODO: Formalize recommendations (not just a string, but classes)

        if solver.max_vars and metadata.qubo_size > solver.max_vars:
            metadata.recommendations.append(
                "Solver %s has inadequate QuBits (%d) for this QUBO "
                "(%d variables)."
                % (solver.name, solver.max_vars, metadata.qubo_size)
            )
            return False

        if solver.max_connectivity and \
                metadata.connectivity[1] > solver.max_connectivity:
            metadata.recommendations.append(
                "Solver %s allows less connectivity (%d) than needed for "
                "this QUBO (%s)." % (
                    solver.name,
                    solver.max_connectivity,
                    str(metadata.connectivity)
                )
            )
            return False

        return True

    def _compute_best_solver_idx(self, metadata):
        """Computes the index of the best solver for the given QUBO (inplace).

        The best solver is currently defined as the one with the least running
        time required.
        """
        # NOTE: The best solver index (bset_solver_idx) includes disabled
        # solvers right now for consistency reasons. The other option would
        # result in indices denoting different solvers in different
        # experiments. This may or may not be desired (for instance, with
        # disabled solvers, class imbalance in neural network classification
        # would return wrong values). Something to keep in mind for the future.
        energies = []
        runtimes = []

        solver_performances = []

        for solver_name in self.solvers:
            if solver_name not in metadata.runtimes:
                solver_performances.append(np.nan)
                continue

            if self.solver_performance_metric == 'average':
                performance_value = np.average(metadata.energies[solver_name])
                performance_value += np.average(metadata.runtimes[solver_name])
            elif self.solver_performance_metric == 'best':
                performance_value = metadata.energies[solver_name][0][0]
                performance_value += metadata.runtimes[solver_name][0]

            solver_performances.append(performance_value)

            energies.append(np.average(metadata.energies[solver_name]))
            runtimes.append(np.average(metadata.runtimes[solver_name]))

        metadata.solver_performance = solver_performances
        metadata.best_solver_idx = np.nanargmin(solver_performances)

        self.logger.debug(solver_performances)
        self.logger.debug(energies)
        self.logger.debug(runtimes)
        self.logger.debug(
            "BEST SOLVER: %s %d" % (
                self.solvers[metadata.best_solver_idx],
                metadata.best_solver_idx
            )
        )

    def intercept_single_proc(self, metadata):
        """Called by ``Interceptor.intercept``, runs single process interception.
        """
        self.enrich_metadata(metadata)

        for solver_name in self.solvers:
            solver = SOLVERS[solver_name]

            if not solver.enabled:
                continue

            #self.logger.info(solver_name)

            if not self.is_adquate_solver(solver, metadata):
                continue

            metadata.runtimes[solver_name] = []
            metadata.solutions[solver_name] = []
            metadata.energies[solver_name] = []

            for _ in range(solver.repeats):
                start_time = time.time()
                try:
                    sol, energy_ = solver(metadata.Q)
                except:
                    traceback.print_exc()
                    continue
                metadata.runtimes[solver_name].append(time.time() - start_time)
                energy = [sol[0].transpose() @ metadata.Q @ sol[0]]
                if self.wrong_energy_check and energy_[0] != energy[0]:
                    raise Exception(
                        "Solver " + str(solver_name) +
                        " returned wrong energy: " + str(energy_[0]) + " != " +
                        str(energy[0])
                    )
                metadata.solutions[solver_name].append(sol)
                metadata.energies[solver_name].append(energy)

        self._compute_best_solver_idx(metadata)

    def intercept_multi_proc(self, metadata):
        """Called by ``Interceptor.intercept``, runs multi process interception.
        """
        self.enrich_metadata(metadata)

        mgr = multiprocessing.Manager()
        sync_dict = mgr.dict()
        procs = []

        for solver_name in self.solvers:
            solver = SOLVERS[solver_name]

            if not solver.enabled:
                continue

            #self.logger.info(solver_name)

            sync_dict[solver_name] = mgr.dict()
            sync_dict[solver_name]["runtimes"] = mgr.list()
            sync_dict[solver_name]["solutions"] = mgr.list()
            sync_dict[solver_name]["energies"] = mgr.list()

            if not self.is_adquate_solver(solver, metadata):
                continue

            for _ in range(solver.repeats):
                p = multiprocessing.Process(
                    target=compute_qubo_multiproc,
                    args=(
                        solver,
                        solver_name,
                        metadata.Q,
                        sync_dict,
                        self.wrong_energy_check
                    )
                )
                p.start()
                procs.append(p)

        i = 0
        for p in procs:
            # self.logger.info(i)
            i += 1
            p.join()

        for solver_name in self.solvers:
            solver = SOLVERS[solver_name]

            if not solver.enabled:
                continue

            metadata.runtimes[solver_name] = list(
                sync_dict[solver_name]["runtimes"])
            metadata.solutions[solver_name] = list(
                sync_dict[solver_name]["solutions"])
            metadata.energies[solver_name] = list(
                sync_dict[solver_name]["energies"])

        self._compute_best_solver_idx(metadata)

    def intercept(self, metadata, use_multi_proc=None):
        """Intercept a QUBO, analyze and modify it, and execute configured
        solvers.

        Args:
            metadata: The Metadata object that contains the QUBO.
            use_multi_proc: Whether to use multiple processes for the solvers.
                Either ``Interceptor.intercept_multi_proc`` or
                ``Interceptor.intercept_single_proc`` are called depending on
                the value of this argument.
                By default this parameter is set to None, at which point the
                setting of the same name from the configuration (config.json)
                is used. The parameter set in this function takes precedence
                over the one in the configuration.

        The Metadata object that contains the QUBO is modified inplace and
        nothing is returned.
        """
        do_use_multi_proc = self.use_multi_proc
        if use_multi_proc is not None:
            do_use_multi_proc = use_multi_proc

        if do_use_multi_proc:
            self.intercept_multi_proc(metadata)
        else:
            self.intercept_single_proc(metadata)

    def compute_connectivity(self, metadata):
        """Compute the min/max/average connectivity of the QUBO.

        Connectivity is defined as the minimum, maximum and average number of
        edges connected to any of the vertices, if the QUBO is interpreted as a
        graph.
        """
        Q = np.triu(metadata.Q)
        counts = np.count_nonzero(Q, axis=1)
        return (counts.min(), counts.max(), counts.mean())

    def compute_density(self, metadata):
        """Compute the graph density of the QUBO if the non-zero entries are
        seen as nodes.
        """
        Q = metadata.Q
        # count every nonezero entry in the upper right
        # don't count linear terms, where x = y
        count_edges = 0
        for (x,y), value in np.ndenumerate(Q):
            if value != 0:
                if x < y:
                    count_edges += 1
        shape = Q.shape[0]
        max_edges = shape * (shape - 1) / 2
        if max_edges == 0:
            density = 0
        else:
            density = count_edges / max_edges
        return density

    def compute_distribution(self, metadata):
        """Compute the mean and variance of the QUBO values.
        """
        Q = metadata.Q
        return (np.average(Q), np.var(Q))

    def enrich_metadata(self, metadata):
        """Enrich the Metadata object by computing additional features, such as
        the size of the QUBO or the min/max/average connectivity of the QUBO.

        Modifies the Metadata object inplace and returns nothing.
        """
        metadata.connectivity = self.compute_connectivity(metadata)
        metadata.density = self.compute_density(metadata)
        metadata.qubo_size = metadata.Q.shape[0]
        metadata.distribution = self.compute_distribution(metadata)
        # TODO Enrich some more..!
