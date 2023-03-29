from abc import ABC

import numpy as np
from numpy import random

from evolution_new.new_visualisation import plot_points, qubo_heatmap
from transformator.common.util import distance_matrix_to_directed_graph
from transformator.generalizations.graph_based.include_graph_structure import include_graph_structure
from transformator.problems.problem import Problem

critical_disorder = 1.07
scaling_parameter = 0.43


def get_random_node_number(size: tuple[int, int]) -> int:
    rng = random.default_rng()
    return rng.integers(size[0], size[1] + 1)


def get_cities(N: int) -> list[list[float, float]]:
    random_disorder = get_random_disorder_parameter(N)
    # random_disorder = 0
    return [apply_disorder_to_city(random_disorder, city) for city in get_cities_on_circle(N)]
    #return [[0., 0.], [1., 1.], [2., -1.], [3., 0.], [4., -1.]]


def get_cities_on_circle(N: int) -> list[list[float, float]]:
    city_coordinates = []
    radius = N / (2 * np.pi)
    for i in range(N):
        phi = (i / N) * (2 * np.pi)
        # city_coordinates.append([radius * np.cos((2 * np.pi) / (i + 1)), radius * np.sin((2 * np.pi) / (i + 1))])
        city_coordinates.append([radius * np.cos(phi), radius * np.sin(phi)])
    # print(city_coordinates)
    # plot_points(city_coordinates)
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
        random_disorder = rng.normal(critical_disorder, 2 * (critical_disorder - min_value))
    return random_disorder / np.power(N, scaling_parameter) + critical_disorder


def get_distance_matrix(N: int, city_coordinates: list[list[float, float]]) -> np.ndarray:  # list[list[float]]:
    # plot_points(city_coordinates)
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            dist = np.sqrt(np.sum(np.subtract(city_coordinates[i], city_coordinates[j]) ** 2))
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return dist_matrix


class TSPQuadrants(Problem, ABC):
    def __init__(self, cfg, dist_matrix, cities, tsp=True, P=10):
        self.dist_matrix = dist_matrix
        self.P = P

    def gen_qubo_matrix(self):
        n = len(self.dist_matrix)

        Q = np.zeros((n ** 2, n ** 2))

        U = - self.P * np.eye(n) + 2 * self.P * np.triu(np.ones((n, n)), k=1)

        # add unary constraint to QUBO
        for i in range(n):
            Q[i * n:i * n + n, i * n:i * n + n] += U

        # Set up visitation constraint and add to QUBO
        for i in range(n - 1, 0, -1):
            V = np.diag([self.P] * n * i, k=(n - i) * n)
            Q += V


        # Setting up distance submatrix
        # D = np.zeros((n_city,n_city))
        # for r in range(len(dist)):
        #    D[cnc[r][0],cnc[r][1]] += dist[r]
        # D = D + D.T

        # add distance submatrices to QUBO
        #for i in range(n - 1):
        #    print(i)
        #    Q[i * n: i * n + n, (i + 1) * n: (i + 1) * n + n] += self.dist_matrix
        #qubo_heatmap(Q)
        # for the trip back home from the last city
        #Q[:n, -n:] += self.dist_matrix
        for i in range(n):  # City 1
            for j in range(i):  # City 2
                dist = self.dist_matrix[i][j]
                for k in range(n - 1):  # Fill sub diagonal in city-city quadrant with distance
                    Q[j * n + k][i * n + k + 1] = dist
                    Q[j * n + k + 1][i * n + k] = dist
                Q[j * n + (n - 1)][i * n] = dist  # Distance between last and first
                Q[j * n][i * n + (n - 1)] = dist
        D = np.diag(np.diagonal(Q))
        Q += Q.T
        Q -= D
        #qubo_heatmap(Q)
        #print(Q)
        return Q

    @classmethod
    def gen_problems(self, cfg, n_problems, size=(4, 4), **kwargs):
        problems = []
        for _ in range(n_problems):
            N = get_random_node_number(size)
            cities = get_cities(N)
            dist_matrix = get_distance_matrix(N, cities)
            #print('Dist matrix: ', dist_matrix)
            problems.append({"dist_matrix": dist_matrix.tolist(), 'cities': cities, 'tsp': True})
        return problems
