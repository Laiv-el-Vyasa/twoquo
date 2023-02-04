import itertools
import numpy as np
from transformator.problems.problem import Problem
from transformator.common.util import gen_subsets_matrix
from transformator.common.util import gen_subset_problems


class ExactCover(Problem):
    def __init__(self, cfg, subset_matrix, A=2, B=2):
        #print('SubsetMatrix: ' + str(subset_matrix))
        self.subset_matrix = subset_matrix
        n = self.subset_matrix.shape[1]
        self.A = n * B + 1
        self.B = B

    def gen_qubo_matrix2(self):
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

    def gen_qubo_matrix(self):
        subsets = np.array(self.subset_matrix)
        print('subsets: ', subsets)
        M = subsets.shape[0]
        N = subsets.shape[1]
        print(M, N)

        A = np.zeros((M, M))
        B = np.zeros((N, M))
        C = np.zeros(M)

        # Fill the matrix A
        for i in range(M):
            for j in range(i+1, M):
                common = 0
                for k in range(N):
                    common += subsets[i][k] * subsets[j][k]
                A[i][j] = common
                A[j][i] = common

        # Fill the matrix B
        for i in range(N):
            for j in range(M):
                B[i][j] = subsets[j][i]

        # Define the QUBO
        Q = np.outer(C, C) + np.dot(np.dot(B, A), B.T)
        return Q

    @classmethod
    def gen_matrix(cls, set_, subsets):
        return gen_subsets_matrix(ExactCover, set_, subsets)

    @classmethod
    def gen_problems(cls, cfg, n_problems, size=(20, 25), **kwargs):
        return gen_subset_problems(ExactCover, "EC", cfg, n_problems, size, **kwargs)
