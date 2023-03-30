import itertools
import numpy as np
from numpy import random

from evolution_new.new_visualisation import qubo_heatmap
from transformator.problems.problem import Problem
from transformator.common.util import gen_subsets_matrix
from transformator.common.util import gen_subset_problems
import pyqubo
from pyqubo import *


def get_element_number(size: tuple[int, int]) -> int:
    rng = random.default_rng()
    return rng.integers(size[0], size[1])


def get_element_subset_factor(size: list[float, float], center: float) -> float:
    rng = random.default_rng()
    std_deviation = np.mean([np.abs(center - i) for i in size])
    r = 0.
    while not size[0] < r < size[1]:
        r = rng.normal(center, std_deviation)
    return r


def get_subset_number(m: int, r: float) -> int:
    subsets = int(m / r)
    while subsets > 3 * m:
        subsets -= 1
    return subsets


# List of elements with subsets they are represented in
def get_subset_matrix(elements: int, subsets: int) -> list[list[int]]:
    # Create list with three times all elements
    # print('m: ', elements, 'n: ', subsets)
    element_list = []
    for i in range(elements):
        element_list.extend([i, i, i])  # Every Element in three subsets
    random.shuffle(element_list)

    # Create subset_list
    subset_list = [[] for _ in range(subsets)]
    subset_matrix = np.zeros((elements, subsets))

    # Fill each subset with at least one element
    for i in range(subsets):
        j = element_list.pop(0)
        subset_matrix[j][i] = 1

    # Distribute to remaining elements on subsets
    rng = random.default_rng()
    for element in element_list:
        i = rng.integers(0, subsets)
        while subset_matrix[element][i] == 1:
            i = rng.integers(0, subsets)
        subset_matrix[element][i] = 1

    return subset_matrix


# List of subsets and the elements they contain
def get_subset_matrix2(elements: int, subsets: int) -> list[list[int]]:
    # Create list with three times all elements
    print('m: ', elements, 'n: ', subsets)
    element_list = []
    for i in range(elements):
        element_list.extend([i, i, i])  # Every Element in three subsets
    random.shuffle(element_list)

    # Create subset_list
    subset_list = [[] for _ in range(subsets)]
    subset_matrix = np.zeros((subsets, elements))

    # Fill each subset with at least one element
    for i in range(subsets):
        j = element_list.pop(0)
        subset_matrix[i][j] = 1

    # Distribute to remaining elements on subsets
    rng = random.default_rng()
    for element in element_list:
        i = rng.integers(0, subsets)
        while subset_matrix[i][element] == 1:
            i = rng.integers(0, subsets)
        subset_matrix[i][element] = 1

    return subset_matrix


class ExactCover(Problem):
    def __init__(self, cfg, subset_matrix, A=2, B=2):
        #print('SubsetMatrix: ' + str(subset_matrix))
        self.subset_matrix = subset_matrix
        n = self.subset_matrix.shape[1]
        self.A = n * B + 1
        self.B = B

    def gen_qubo_matrix(self):
        # get_pygubo_matrix(self.subset_matrix, self.subset_matrix.shape[0], self.subset_matrix.shape[1])
        # print('subsets: ',  self.subset_matrix)
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
        #qubo_heatmap(Q)
        return Q

    @classmethod
    def gen_matrix(cls, set_, subsets):
        return gen_subsets_matrix(ExactCover, set_, subsets)

    @classmethod
    def gen_problems(cls, cfg, n_problems, size=(16, 64), **kwargs):
        target = cfg["problems"]["EC"].get("target_factor", 0.62)
        target_boundary = cfg["problems"]["M3SAT"].get("target_boundary", [0.34, 2.0])
        problems = []
        for i in range(n_problems):
            m = get_element_number(size)
            n = get_subset_number(m, get_element_subset_factor(target_boundary, target))
            problems.append(get_subset_matrix(m, n))
        return [{"subset_matrix": matrix} for matrix in problems]
