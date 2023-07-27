import random
import itertools

import numpy as np
from numpy import random as np_random

from evolution_new.new_visualisation import qubo_heatmap
from transformator.problems.problem import Problem


def get_random_var_number(cfg: dict) -> int:
    size = cfg["problems"]["M3SAT"].get("size_vars", [16, 32])
    rng = np_random.default_rng()
    return rng.integers(size[0], size[1] + 1)


def gen_random_clause_number(cfg: dict, n_vars: int) -> int:
    target = cfg["problems"]["M3SAT"].get("target_density", 4)
    desity_size = cfg["problems"]["M3SAT"].get("density_size", [0, 8])
    rng = np_random.default_rng()
    desity = 0
    while desity <= desity_size[0] or desity >= desity_size[1]:
        desity = rng.normal(target, (desity_size[1] - desity_size[0]) / 4)
    return int(np.round(desity * n_vars))


def get_clause(var_list: list[int]) -> tuple[tuple[int, bool], tuple[int, bool], tuple[int, bool]]:
    random.shuffle(var_list)
    clause_list = [(var_list[0], bool(random.getrandbits(1))), (var_list[1], bool(random.getrandbits(1))),
                   (var_list[2], bool(random.getrandbits(1)))]
    clause_list = sorted(clause_list, key=lambda x: (not x[1], x[0]))
    return clause_list[0], clause_list[1], clause_list[2]


def analyze_clause(clause: tuple[tuple[int, bool], tuple[int, bool], tuple[int, bool]]) -> tuple[list[int], str]:
    pattern = [1, 1, 1]
    for i in range(3):
        if not clause[i][1]:
            pattern[i] = 0
    string = ''
    if pattern == [0, 0, 0]:
        string = 'n' + str(clause[0][0]) + '_n' + str(clause[1][0]) + '_n' + str(clause[2][0])
    elif pattern == [1, 0, 0]:
        string = str(clause[0][0]) + '_n' + str(clause[1][0])
    else:
        string = str(clause[0][0]) + '_' + str(clause[1][0])
    return pattern, string


class Max3SAT(Problem):
    def __init__(self, cfg, clauses, n_vars):
        self.clauses = clauses
        self.n_vars = n_vars

    def gen_qubo_matrix(self):
        n = self.n_vars + len(self.clauses)
        Q = np.zeros((n, n))
        clause_dict = {}
        next_free_clause_idx = self.n_vars

        for i, c in enumerate(self.clauses):
            clause_pattern, clause_string = analyze_clause(c)
            # print(clause_pattern)
            c_idx = next_free_clause_idx
            if clause_string in clause_dict.keys():
                c_idx = clause_dict[clause_string]
            else:
                next_free_clause_idx += 1
                clause_dict[clause_string] = c_idx

            if clause_pattern == [1, 1, 1]:
                Q[c[0][0], c[1][0]] += 2
                Q[c[0][0], c_idx] -= 2
                Q[c[1][0], c_idx] -= 2
                Q[c[2][0], c[2][0]] -= 1
                Q[c[2][0], c_idx] += 1
                Q[c_idx, c_idx] += 1
            elif clause_pattern == [1, 1, 0]:
                Q[c[0][0], c[1][0]] += 2
                Q[c[0][0], c_idx] -= 2
                Q[c[1][0], c_idx] -= 2
                Q[c[2][0], c[2][0]] += 1
                Q[c[2][0], c_idx] -= 1
                Q[c_idx, c_idx] += 2
            elif clause_pattern == [1, 0, 0]:
                Q[c[0][0], c[0][0]] += 2
                Q[c[0][0], c[1][0]] -= 2
                Q[c[0][0], c_idx] -= 2
                Q[c[1][0], c_idx] += 2
                Q[c[2][0], c[2][0]] += 1
                Q[c[2][0], c_idx] -= 1
            else:
                Q[c[0][0], c[0][0]] -= 1
                Q[c[0][0], c[1][0]] += 1
                Q[c[0][0], c[2][0]] += 1
                Q[c[0][0], c_idx] += 1
                Q[c[1][0], c[1][0]] -= 1
                Q[c[1][0], c[2][0]] += 1
                Q[c[1][0], c_idx] += 1
                Q[c[2][0], c[2][0]] -= 1
                Q[c[2][0], c_idx] += 1
                Q[c_idx, c_idx] -= 1
            # print(c)
            # qubo_heatmap(Q)
        # qubo_heatmap(Q)
        # qubo_heatmap(Q.T)
        # qubo_heatmap(Q.T + Q)
        Q = Q + Q.T - np.diag(np.diag(Q))
        # qubo_heatmap(Q)
        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size=(10, 128), **kwargs):
        # size: First is n_vars, second is number of clauses.
        # Want something like this:
        # [((1, True), (2, True), (3, True)), ((1, True), (2, False), (4, False))]

        problems = []
        n_vars_list = []
        for _ in range(n_problems):
            problem = []
            clause_set = set()
            n_vars = get_random_var_number(cfg)
            n_vars_list.append(n_vars)
            var_list = [n for n in range(n_vars)]
            clauses = gen_random_clause_number(cfg, n_vars)
            # Eg: 25 clauses
            for _ in range(clauses):
                clause = get_clause(var_list)
                while clause in clause_set:
                    clause = get_clause(var_list)
                clause_set.add(clause)
                problem.append(clause)
            if len(problem) == clauses:
                problems.append(sorted(problem))
        len_ = len(problems)

        return [{"clauses": problem, "n_vars": n_vars} for problem, n_vars in zip(problems, n_vars_list)]
