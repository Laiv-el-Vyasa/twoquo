"""Contains the Metadata class and the Database interface.
"""
import io
import gzip
import copy
import pickle
from abc import ABCMeta
from abc import abstractmethod

from transformator.transformator import to_compact_matrix
from database.serialize_util import gen_qubo_key


class Metadata:
    """This class holds metadata for a QUBO or problem instance.

    This can also be compared to a single row in the database.

    Attributes:
        Q: The QUBO matrix.
        Q_prime: The QUBO matrix that was modified by the interceptor.
        qubo_size: The size of one dimension of the QUBO matrix, i.e. the
            number of decision variables. For a 64x64 QUBO, the size would be
            64.
        solutions: The map of solutions, with the key being the solver, and the
            value being the best 20 solution vectors of that solver, with n
            (default: 10) repeat calls (i.e. in total 10 * 20 solution
            vectors).
            Example (2 out of 10 repeat calls shown):

                [[0, 0, 0, ..., 1, 1, 0],
                 [1, 0, 0, ..., 1, 1, 0],
                 [0, 0, 0, ..., 1, 0, 0],
                 ...,
                 [0, 0, 0, ..., 1, 1, 0],
                 [0, 0, 0, ..., 0, 0, 0],
                 [1, 0, 0, ..., 1, 0, 0]],
                [[0, 1, 1, ..., 1, 1, 0],
                 [1, 1, 0, ..., 0, 1, 0],
                 [0, 1, 1, ..., 0, 1, 0],
                 ...,
                 [0, 1, 0, ..., 1, 0, 1],
                 [0, 1, 0, ..., 0, 1, 0],
                 [1, 0, 1, ..., 0, 1, 1]]
                ...

        energies: The map of energies, with the key being the solver, and the
            value being the best 20 energies (corresponding to the solution
            vectors) of the solver, with n (default: 10) repeat calls.
        runtimes: The map of runtimes, with the key being the solver, and the
            value being the total runtime in seconds needed by that solver,
            with n (default: 10) repeat calls.
            Example:

                {
                    'simulated_annealing’: [
                        7.2242348194122314, 8.062700271606445, 9.022322177886963,
                        9.113363027572632, 9.292305707931519, 9.836304187774658,
                        10.236612319946289, 10.325115442276001, 10.911486625671387,
                        12.034629106521606
                    ],
                    'tabu_search': [
                        4.200134038925171, 4.63025689125061, 5.6781463623046875,
                        5.81449294090271, 6.078419923782349, 6.179047584533691,
                        6.231500148773193, 6.972151041030884, 9.067768335342407,
                        9.309316158294678
                    ]
                }

        qubo_type: The problem type (e.g. traveling_salesman,
            number_partitioning).
        recommendations: A set of strings denoting recommendations for the
            given problem instance or QUBO type. This could for instance
            include a hardware or algorithm recommendation (i.e., the optimal
            algorithm in terms of expected solution quality or runtime).
        connectivity: A triple consisting of the minimum, maximum and average
            degree of the QUBO, if represented as a graph. Since some Quantum
            Hardware only supports a certain maximum connectivity (i.e. a QuBit
            is only connected to a certain maximum number of other QuBits),
            this may aid in selection of suitable hardware/algorithms. Note
            that when the connectivity of the Hardware is inadequate, hybrid
            methods may help.
        density: If the QUBO is represented as a graph, with non-zero entries
            seen as nodes in the graph, this gives the density of that graph.

    TODO: Mention the naming convention of the qubo_type field.
    """
    def __init__(self):
        self.Q = None
        self.Q_prime = None
        self.qubo_size = None
        self.approx_steps = None
        self.approx_strategy = None
        self.solutions = {}
        self.approx_solution_quality = {} #Dict of approximation step with dict of solver and solution quality
        self.problem = None
        self.energies = {}
        self.runtimes = {}
        self.qubo_type = None
        self.recommendations = []
        self.connectivity = None
        self.density = None

    def key(self):
        """Returns a unique identifier for the QUBO.

        Specifically, returns the output of
        ``tooquo.database.serialize_util.gen_qubo_key``.
        """
        return gen_qubo_key(self.Q)

    def serialize(self):
        """Serialize this object for database entry.

        Before saving in the database, the large QUBO matrices are converted
        to a compact view using
        ``tooquo.transformator.transformator.to_compact_matrix``.

        Note: This function is not atomic and not thread-safe.

        Returns this object serialized as bytes (uses pickle).
        """
        Q = copy.deepcopy(self.Q)
        Q_prime = copy.deepcopy(self.Q_prime)
        recommendations = copy.deepcopy(self.recommendations)

        self.Q = to_compact_matrix(self.Q)[0]
        self.Q_prime = to_compact_matrix(self.Q_prime)[0]
        del self.recommendations

        gzip_output = io.BytesIO()
        g = gzip.GzipFile(fileobj=gzip_output, mode='w')
        g.write(pickle.dumps(self))
        g.close()
        gzip_output.seek(0)

        self.Q = Q
        self.Q_prime = Q_prime
        self.recommendations = recommendations

        return gzip_output

    def __repr__(self):
        return "|".join(
            str(x) for x in [
                self.Q,
                self.Q_prime,
                self.qubo_size,
                self.approx_steps,
                self.approx_strategy,
                self.solutions,
                self.approx_solution_quality,
                self.problem,
                self.energies,
                self.runtimes,
                self.qubo_type,
                self.recommendations
            ]
        )


class Database:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def init(self):
        """Init the underlying database object, e.g. MongoClient or LMDBLoader.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close the underlying database object."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata_by_qubo(self, Q):
        """Get a Metadata object by QUBO Q (vector of flattened QUBO)."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata_by_problem(self, problem_def):
        """Get a Metadata object by problem definition (list of vectors).

        The list of vectors can represent flat matrices (e.g. adjacency
        matrix).
        """
        raise NotImplementedError

    @abstractmethod
    def save_metadata(self, metadata):
        """Save a MetaData object from the Monitor.

        The original input is either a QUBO or a problem definition."""
        raise NotImplementedError

    @abstractmethod
    def iter_metadata(self):
        """Iterate through all Metadata items in the database."""
        raise NotImplementedError

    @abstractmethod
    def size(self):
        """Return the size of the database."""
        raise NotImplementedError
