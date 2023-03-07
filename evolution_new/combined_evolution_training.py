from typing import Callable

from training_config import training_config, learning_parameters_config, model_config, \
    fitness_function_generation_config
from config import load_cfg
from pygad_learning import PygadLearner
from learning_model import LearningModel
from evolution_utils import fitness_params_to_string


def get_data_from_training_config(tr_config_name: str) -> tuple[LearningModel, dict, Callable[[dict, dict], float]]:
    # Select training config for training
    tr_config = training_config[tr_config_name]

    # Extract all necessary information from config
    config_name = tr_config['config_name']
    training_name = tr_config['training_name']
    config = load_cfg(cfg_id=config_name)
    problem = config["pipeline"]["problems"]["problems"][0]
    size = config['pipeline']['problems']['qubo_size']
    fitness_parameters = tr_config['fitness_parameters']
    training_name = f'{training_name}_{problem}_{size}{fitness_params_to_string(fitness_parameters)}'

    # Get fitness function
    fitness_function = fitness_function_generation_config[tr_config['fitness_function']](fitness_parameters)

    # Get network model
    model = model_config[tr_config['network_type']](tr_config['network_information'])

    # Get learning parameters
    learning_parameters = learning_parameters_config[tr_config['learning_parameters']]
    learning_parameters['config_name'] = config_name
    learning_parameters['training_name'] = training_name

    return model, learning_parameters, fitness_function


# Construct pygad learner
if __name__ == "__main__":
    config_name = 'combined_npp'
    model, learning_parameters, fitness_func = get_data_from_training_config(config_name)
    pygad_learner = PygadLearner(model, learning_parameters, fitness_func)
    pygad_learner.learn_model()
