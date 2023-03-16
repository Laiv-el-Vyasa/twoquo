from abc import ABC

import numpy as np
from numpy import random

from transformator.common.util import distance_matrix_to_directed_graph
from transformator.generalizations.graph_based.include_graph_structure import include_graph_structure
from transformator.problems.problem import Problem

critical_disorder = 1.07
scaling_parameter = 0.43


def get_random_node_number(size: tuple[int, int]) -> int:
    rng = random.default_rng()
    return rng.integers(size[0], size[1])


def get_cities_on_circle(N: int) -> list[list[float, float]]:
    city_coordinates = []
    radius = N / (2 * np.pi)
    for i in range(N):
        city_coordinates.append([radius * np.cos((2 * np.pi) / i), radius * np.sin((2 * np.pi) / i)])
    return city_coordinates


def apply_disorder_to_city(disorder_parameter: float, city_coordinates: list[float, float]) -> list[float, float]:
    rng = random.default_rng()
    random_radius = rng.uniform(0, disorder_parameter)
    random_angle = rng.uniform(0, 2 * np.pi)
    city_disorder = [random_radius * np.cos(random_angle), random_radius * np.sin(random_angle)]
    return np.add(city_coordinates, city_disorder)


def get_random_disorder_parameter(N: int) -> float:
    min_value = (0 - critical_disorder) * np.power(N, scaling_parameter)
    max_value = 40
    rng = random.default_rng()
    random_disorder = min_value - 1
    while not min_value < random_disorder < max_value:
        rng.normal(critical_disorder, 2 * (critical_disorder - min_value))
    return random_disorder / np.power(N, scaling_parameter) + critical_disorder


def get_distance_matrix(N: int) -> np.ndarray:  # list[list[float]]:
    random_disorder = get_random_disorder_parameter(N)
    city_coordinates = [apply_disorder_to_city(random_disorder, city) for city in get_cities_on_circle(N)]
    dist_matrix = np.zeros(N)
    for i in range(N):
        for j in range(i):
            dist = np.sqrt(np.sum(np.subtract(city_coordinates[i], city_coordinates[j]) ** 2, axis=1))
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return dist_matrix


class TSPQuadrants(Problem, ABC):
    def __init__(self, cfg, dist_matrix, P=10):
        self.dist_matrix = dist_matrix
        self.P = 10

    def gen_qubo_matrix(self):
        n = len(self.dist_matrix)
        Q = np.zeros((n ** 2, n ** 2))

        # quadrants_y = list(range(0, n ** 2, n))
        # quadrants_x = quadrants_y[1:] + [quadrants_y[0]]
        #
        # # The diagonal positive constraints
        # for start_x in quadrants_y:
        #     for start_y in quadrants_y:
        #         for i in range(n):
        #             Q[start_x + i][start_y + i] = 2 * self.constraint
        #
        # # The distance matrices
        # for (start_x, start_y) in zip(quadrants_x, quadrants_y):
        #     for i in range(n):
        #         for j in range(n):
        #             if i == j:
        #                 continue
        #             Q[start_x + i][start_y + j] = self.P * self.dist_matrix[j][i]
        #         Q[start_x + i][start_y + i] = 2 * self.constraint
        #
        # # The middle diagonal negative constraints
        # for start_x in quadrants_x:
        #     for i in range(n):
        #         Q[start_x + i][start_x + i] = -2 * self.constraint
        #         for j in range(n):
        #             if i != j:
        #                 Q[start_x + i][start_x + j] += 2 * self.constraint

        include_graph_structure(
            qubo_in=Q,
            graph=distance_matrix_to_directed_graph(self.dist_matrix),
            positions=n  # ,
            # score_edge_weights=1.0
        )

        print(Q)

        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size=(4, 4), **kwargs):
        # TODO: 200 and 100 (contraint) is hardcoded !!
        problems = []
        for _ in range(n_problems):
            N = get_random_node_number(size)
            dist_matrix = get_distance_matrix(N)
            print(dist_matrix)
            np.fill_diagonal(dist_matrix, 0)
            problems.append({"dist_matrix": dist_matrix.tolist(), 'tsp': True})
        return problems
