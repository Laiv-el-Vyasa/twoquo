import numpy as np
from numpy import random
from transformator.problems.problem import Problem


def get_problems(cfg, n_problems, size: tuple[int, int], target: float):
    rng = random.default_rng()
    problems = np.empty(shape=n_problems, dtype=np.ndarray)
    #print('empty problems: ', problems)
    for n in range(n_problems):
        number_bound = get_number_bound(cfg)
        numbers = rng.integers(0, number_bound, get_problem_size(size, target))
        #print('numbers: ', numbers)
        problems[n] = numbers
        #np.append(problems, numbers)
    #problems = rng.geometric(1 / (size / 4), size=(n_problems, size)) * \
    #           random.randint(0, size, (n_problems, size)) + \
    #           random.randint(0, size, (n_problems, size))
    #problems = np.array(problems)
    #print('problems: ', problems)
    return problems
    #return problems.astype(int)


def get_problem_size(size: tuple[int, int],
                     target: float):
    rng = random.default_rng()
    problem_size = 0
    while problem_size < size[0] or problem_size > size[1]:
        problem_size = rng.normal(target, ((size[1] - size[0]) / 4))
    return int(np.round(problem_size))


def get_number_bound(cfg):
    high = np.power(2, cfg['problems']['NPP']['bits'])
    return high


class NumberPartitioning(Problem):
    def __init__(self, cfg, numbers):
        self.numbers = numbers

    def gen_qubo_matrix(self):
        n = len(self.numbers)
        c = sum(self.numbers)

        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    Q[i][j] = self.numbers[i] * (self.numbers[i] - c)
                else:
                    Q[i][j] = self.numbers[i] * self.numbers[j]
        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size=(16, 64), **kwargs):
        problems = get_problems(cfg, n_problems, size, cfg['problems']['NPP']['transition'])
        return [{"numbers": problem} for problem in problems]
