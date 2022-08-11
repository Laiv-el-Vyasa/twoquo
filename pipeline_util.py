import numpy as np

from paths import PATH_DB
from monitor import Monitor
from database.database import Metadata
from database.lmdb_database import LmdbDatabase
#from tooquo.loggin import GLOBAL_LOGGER
from transformator.problems import PROBLEM_REGISTRY
from transformator.problems.number_partitioning import NumberPartitioning


class QUBOGenerator:
    def __init__(self, full_cfg):
        self.full_cfg = full_cfg
        self.cfg = full_cfg["pipeline"]
        self.n_problems = self.cfg['problems']['n_problems']
        self.qubo_size = self.cfg['problems']['qubo_size']
        self.problems = self._prep_problems()
        #self.logger = GLOBAL_LOGGER.configure(full_cfg)

    def _prep_problems(self):
        ret = []
        for name in self.cfg['problems']['problems']:
            ret.append(
                (PROBLEM_REGISTRY[name], self.cfg['problems'][name], name)
            )
        return ret

    def gen_qubo_matrices(self, cls, n_problems, **kwargs):
        problems = cls.gen_problems(self.cfg, n_problems, **kwargs)
        qubo_matrices = [
            cls(self.cfg, **problem).gen_qubo_matrix()
            for problem in problems
        ]
        return problems, qubo_matrices

    def generate(self):
        all_problems = []
        data = []
        labels = []

        for i, (cls, kwargs, name) in enumerate(self.problems):
            problems, qubo_matrices = self.gen_qubo_matrices(
                cls, self.n_problems, **kwargs
            )

            all_problems.extend(problems)
            qubo_matrices = np.array(qubo_matrices)
            #self.logger.info((cls, qubo_matrices.shape))

            #if self.cfg["model"]["norm_div_max"]:
            #   qubo_matrices = qubo_matrices / np.max(np.abs(qubo_matrices))

            data.extend(qubo_matrices)
            labels.extend([i for _ in range(len(qubo_matrices))])

        #self.logger.info(("TOTAL DATA LEN", len(data)))
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return data, labels, all_problems

    def generate_and_save(self, cfg_id):
        db = LmdbDatabase(
            self.full_cfg,
            db_path=PATH_DB + '%s.lmdb' % cfg_id
        )
        db.init()
        monitor = Monitor(self.full_cfg, db)

        qubos, labels, problems = self.generate()

        for qubo in qubos:
            qubo = np.asarray(qubo)
            metadata = Metadata()
            metadata.Q = qubo
            metadata.qubo_size = qubo.shape[0]
            metadata.qubo_type = ""  # TODO
            monitor.save(metadata)

        return qubos, labels, problems


class Pipeline:
    pass
