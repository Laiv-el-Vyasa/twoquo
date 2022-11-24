"""Generates recommendations and manages Monitor and Interceptor access. Main
entry point.
"""
import multiprocessing

import numpy as np
import time

from monitor import Monitor
from solver.solver import SOLVERS
from config import load_cfg
from solver.solver import load_solvers
#from tooquo.learner import load_learners
#from tooquo.learner import LEARNERS
from database.database import Metadata
from database.lmdb_database import LmdbDatabase
from interceptor import Interceptor
#from tooquo.loggin import GLOBAL_LOGGER
#from tooquo.loggin import Logger


class RecommendationEngine:
    def __init__(self, db_path=None, cfg=None, model_cfg_id="1"):
        """Create a RecommendationEngine instance.

        Args:
            db_path: The path to the LMDB database. If set to None, a default
                one is inferred.
            cfg: Optionally a config dictionary loaded from JSON (see
                ``tooquo.config.load_cfg``)
            model_cfg_id: Configuration identifier of the pipeline to use for
                this recommendation call. This allows setting up different
                models. Optional.
        """
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = load_cfg(cfg_id=model_cfg_id)
        load_solvers(self.cfg)

        self.db = LmdbDatabase(self.cfg, db_path)
        self.db.init()

        #load_learners(self.cfg)

        #self.db.init()
        self.monitor = Monitor(self.cfg, self.db)
        self.wrong_energy_check = False

        #self.logger = GLOBAL_LOGGER.configure(self.cfg)

    def get_database(self):
        """Getter for the database instance that is initalized in the
        ``RecommendationEngine``.
        """
        return self.db

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

    def recommend(self, qubo, type_=None, solvers=SOLVERS):
        """Returns the TooQuO recommendation as a Response object.

        This Interceptor will recommend a solver from the list of solvers
        passed, and further metadata such as a set of recommended approximation
        levels. If ommitted, all known solvers in the global solver dictionary
        will be considered by the Interceptor.

        Args:
            qubo: The full (symmetric) QUBO matrix. The lower triangle of the
                QUBO should not be zeroed. Lower triangle is defined as
                numpy.tril.
            type_: The problem type of the QUBO, using joined_lower naming
                convention. For example, Maximum Cut should be passed as
                maximum_cut. Maximum and minimum should always be written
                fully, and not shortened (e.g. max). Optional. If set to None,
                the class will be inferred by a classificator.
            solvers: A list of solver keys, with a single key being a unique
                identifier in the global solvers dictionary.

        Returns:
            The recommendation of TooQuO, as a Metadata object.
        """
        qubo = np.asarray(qubo)

        metadata = Metadata()
        metadata.Q = qubo
        metadata.qubo_size = qubo.shape[0]
        metadata.qubo_type = type_

        #Interceptor(self.cfg, solvers).intercept(metadata)
        for solver_name in solvers:
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
                    #print(sol)
                except:
                    #traceback.print_exc()
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

        #self.monitor.save(metadata)
        return metadata


    def save_metadata(self, metadata):
        self.monitor.save(metadata)


    def train_pipeline_by_learner_name(self, learner_name, continue_model="", cfg_id=None):
        """Train a model in pipeline mode using a learner.

        Note that a learner has a cfg_id attached to it, which defines the
        sizes of the problem classes, and the neural network model.

        Args:
            learner_name: The learner name as configured in the configuration.
                See ``tooquo.learner.learner.load_learners`` for more.
            continue_model: Optional filename of a pre-existing model that
                should be continued with.
                See more at ``tooquo.learner.learner.Learner.init_model``.
            cfg_id: Optional cfg_id that the learner should use.
        """
        #learner = LEARNERS[learner_name]
        #learner.init_learner(supplied_cfg_id=cfg_id)
        #learner.init_model(continue_model=continue_model)
        #learner.train()
