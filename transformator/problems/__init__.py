from transformator.problems.number_partitioning import NumberPartitioning
from transformator.problems.max_cut import MaxCut
from transformator.problems.minimum_vertex_cover import MinimumVertexCover
#from tooquo.transformator.problems.set_packing import SetPacking
from transformator.problems.max2sat import Max2SAT
from transformator.problems.set_partitioning import SetPartitioning
from transformator.problems.graph_coloring import GraphColoring
from transformator.problems.quadratic_assignment import QuadraticAssignment
#from tooquo.transformator.problems.quadratic_knapsack import QuadraticKnapsack
from transformator.problems.max3sat import Max3SAT
from transformator.problems.tsp import TSP
from transformator.problems.tsp_quadrants import TSPQuadrants
from transformator.problems.graph_isomorphism import GraphIsomorphism
from transformator.problems.subgraph_isomorphism import SubGraphIsomorphism
from transformator.problems.max_clique import MaxClique
from transformator.problems.exact_cover import ExactCover
from transformator.problems.binary_integer_linear_programming import BinaryIntegerLinearProgramming
from transformator.problems.max_independent_set import MaxIndependentSet
from transformator.problems.minimum_maximum_matching import MinimumMaximumMatching
from transformator.problems.set_cover import SetCover
from transformator.problems.knapsack_integer_weights import KnapsackIntegerWeights
#from tooquo.transformator.problems.traffic_flow import TrafficFlow
from transformator.problems.job_shop_scheduling import JobShopScheduling
#from tooquo.transformator.problems.portfolio_optimization import PortfolioOptimization
from transformator.problems.longest_path import LongestPath
#from tooquo.transformator.problems.minimum_weighted_vertex_cover import MinimumWeightedVertexCover


PROBLEM_REGISTRY = {
    "NP": NumberPartitioning,
    "MC": MaxCut,
    "MVC": MinimumVertexCover,
    #"SP": SetPacking,
    "M2SAT": Max2SAT,
    "SPP": SetPartitioning,
    "GC": GraphColoring,
    "QA": QuadraticAssignment,
    #"QK": QuadraticKnapsack,
    "M3SAT": Max3SAT,
    "TSP": TSPQuadrants,
    "GI": GraphIsomorphism,
    "SGI": SubGraphIsomorphism,
    "MCQ": MaxClique,
    "EC": ExactCover,
    "BIP": BinaryIntegerLinearProgramming,
    "MIS": MaxIndependentSet,
    "MMM": MinimumMaximumMatching,
    "SC": SetCover,
    "KIW": KnapsackIntegerWeights,
    #"TFO": TrafficFlow,
    "JSS": JobShopScheduling,
    #"PO": PortfolioOptimization,
    "LP": LongestPath,
    #"MWVC": MinimumWeightedVertexCover
}


__pdoc__ = {}
__pdoc__["test"] = False
