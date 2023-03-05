from combined_model import CombinedModel
from evolution_utils import construct_standard_fitness_function

training_config = {
    'combined_m3sat':
        {
            'config_name': 'test_evol_m3sat',
            'training_name': 'combined_model',
            'learning_parameters': 'standard',
            'network_type': 'combined',
            'network_information': {
                'network_name': 'combinedModelUwU',
                'node_features': 8,
            },
            'fitness_function': 'standard',
            'fitness_parameters': {
                'a': 1,
                'b': 0.5,
                'c': 10,
                'd': 0.1,
                'z': 0.05
            }
        },
    'combined_mc':
        {
            'config_name': 'test_evol_mc',
            'training_name': 'combined_model',
            'learning_parameters': 'standard',
            'network_type': 'combined',
            'network_information': {
                'network_name': 'combinedModelUwU',
                'node_features': 8,
            },
            'fitness_function': 'standard',
            'fitness_parameters': {
                'a': 1,
                'b': 0.5,
                'c': 10,
                'd': 0.1,
                'z': 0.05
            }
        },
    'combined_test':
        {
            'config_name': 'test_evol_m3sat_test',
            'training_name': 'combined_model',
            'learning_parameters': 'test',
            'network_type': 'combined',
            'network_information': {
                'network_name': 'combinedModelUwU',
                'node_features': 8,
            },
            'fitness_function': 'standard',
            'fitness_parameters': {
                'a': 1,
                'b': 0.5,
                'c': 10,
                'd': 0.1,
                'z': 0.05
            }
        },
}

model_config = {
    'combined': CombinedModel
}

fitness_function_generation_config = {
    'standard': construct_standard_fitness_function
}

learning_parameters_config = {
    'standard': {
        'population': 100,
        'num_generations': 50,
        'keep_elitism': 5,
        'percent_of_parents_mating': 0.2
    },
    'test': {
        'population': 10,
        'num_generations': 5,
        'keep_elitism': 5,
        'percent_of_parents_mating': 0.2
    }
}
