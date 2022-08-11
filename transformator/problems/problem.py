from abc import ABCMeta
from abc import abstractmethod


class Problem:
    __metaclass__ = ABCMeta

    @abstractmethod
    def gen_qubo_matrix(self):
        """Generate a symmetric QUBO matrix for the given problem instance.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def gen_problems(self, cfg, n_problems, size=20, **kwargs):
        """Generate n_problems problem instances with the given size.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_solution_dict(self, cfg, problem, solution, **kwargs):
        """Generate a nice dictionary from the binary solution vector and the problem
        """
        raise NotImplementedError
