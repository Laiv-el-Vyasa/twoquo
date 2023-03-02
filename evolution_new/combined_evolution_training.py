from training_config import training_config, learning_parameters_config, model_config, \
    fitness_function_generation_config
from config import load_cfg
from pygad_learning import PygadLearner
from evolution_utils import fitness_params_to_string

# Select training config for training
training_config = training_config['combined_m3sat']

# Extract all necessary information from config
config_name = training_config['config_name']
training_name = training_config['training_name']
config = load_cfg(cfg_id=config_name)
problem = config["pipeline"]["problems"]["problems"][0]
size = config['pipeline']['problems']['qubo_size']
fitness_parameters = training_config['fitness_parameters']
training_name = f'{training_name}_{problem}_{size}{fitness_params_to_string(fitness_parameters)}'

# Get fitness function
fitness_function = fitness_function_generation_config[training_config['fitness_function']](fitness_parameters)

# Get network model
model = model_config[training_config['network_type']](training_config['network_information'])

# Get learning parameters
learning_parameters = learning_parameters_config[training_config['learning_parameters']]
learning_parameters['config_name'] = config_name
learning_parameters['training_name'] = training_name

# Construct pygad learner
if __name__ == "__main__":
    pygad_learner = PygadLearner(model, learning_parameters, fitness_function)
    pygad_learner.learn_model()
