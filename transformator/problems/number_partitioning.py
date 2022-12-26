import numpy as np
from numpy import random
from transformator.problems.problem import Problem


def get_problems(n_problems, size):
    rng = random.default_rng()
    problems = []
    #for n in range(n_problems):
    #    numbers = []
    #    for i in range(size):
    #        numbers.append(np.floor(rng.normal(2 * size, size)))
    #        if numbers[i] < 0:
    #            numbers[i] = np.floor(rng.normal(2 * size, 4 * size))
    #        if numbers[i] < 0:
    #            numbers[i] = random.randint(0, size * size * 3)
    #    problems.append(numbers)
    problems = rng.geometric(1 / (size / 4), size=(n_problems, size)) * \
               random.randint(0, size, (n_problems, size)) + \
               random.randint(0, size, (n_problems, size))
    problems = np.array(problems)
    return problems.astype(int)


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
    def gen_problems(self, cfg, n_problems, size=20, **kwargs):
        if not cfg['problems']['NP']['new']:
            problems = np.random.randint(0, 100, (n_problems, size))
        else:
            problems = get_problems(n_problems, size)
        return [{"numbers": problem} for problem in problems]
