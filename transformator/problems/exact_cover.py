import itertools
import numpy as np
from tooquo.transformator.problems.problem import Problem
from tooquo.transformator.common.util import gen_subsets_matrix
from tooquo.transformator.common.util import gen_subset_problems


class ExactCover(Problem):
    def __init__(self, cfg, subset_matrix, A=2, B=2):
        #print('SubsetMatrix: ' + str(subset_matrix))
        self.subset_matrix = subset_matrix
        self.A = A
        self.B = B

    def gen_qubo_matrix(self):
        n = self.subset_matrix.shape[1]
        Q = np.zeros((n, n))

        for i in range(n):
            Q[i][i] -= self.A

        # From Lucas 2014: The second term, to find the minimum exact cover:
        for i in range(n):
            Q[i][i] += self.B

        for row in self.subset_matrix:
            idx = list(zip(*np.where(row > 0)))
            tuples = itertools.combinations(idx, 2)
            for j, k in tuples:
                Q[j][k] += self.A / 2
                Q[k][j] += self.A / 2

            if len(idx) == 1:
                Q[idx[0]][idx[0]] -= self.A

        return Q

    @classmethod
    def gen_matrix(cls, set_, subsets):
        return gen_subsets_matrix(ExactCover, set_, subsets)

    @classmethod
    def gen_problems(cls, cfg, n_problems, size=(20, 25), **kwargs):
        return gen_subset_problems(ExactCover, "EC", cfg, n_problems, size, **kwargs)
