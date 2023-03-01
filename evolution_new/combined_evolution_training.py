from combined_model import CombinedModel
from config import load_cfg
from pygad_learning import PygadLearner
from evolution_utils import fitness_params_to_string

config_name = 'test_evol_m3sat'
config = load_cfg(config_name)
fitness_parameters = {
    'a': 1,
    'b': 0.5,
    'c': 10,
    'd': 0.1,
    'z': 0.05
}
problem = config["pipeline"]["problems"]["problems"][0]
size = config['pipeline']['problems']['qubo_size']
training_name = f'combined_{problem}_{size}{fitness_params_to_string(fitness_parameters)}'

