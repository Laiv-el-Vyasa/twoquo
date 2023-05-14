from combined_model import CombinedModel
from combined_feature_model import CombinedFeatureModel
from evolution_new.combined_model_features_onehot import CombinedOneHotFeatureModel
from evolution_new.combined_scale_model import CombinedScaleModel
from evolution_utils import construct_standard_fitness_function, construct_scale_fitness_function

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
    'combined_ec':
        {
            'config_name': 'test_evol_ec',
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
    'combined_ec_scale':
        {
            'config_name': 'test_evol_ec',
            'training_name': 'combined_scale_model',
            'learning_parameters': 'standard',
            'network_type': 'combined_scale',
            'network_information': {
                'network_name': 'combinedScaleModelUwU',
                'node_features': 8,
            },
            'fitness_function': 'scale',
            'fitness_parameters': {
                'a': 1,
                'b': 2,
                'c': 4,
                'd': 0.1,
                'z': 0.25
            }
        },
    'combined_sgi':
        {
            'config_name': 'test_evol_sgi',
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
    'combined_gc':
        {
            'config_name': 'test_evol_gc',
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
    'combined_gc_2':
        {
            'config_name': 'test_evol_gc',
            'training_name': 'combined_model_load',
            'learning_parameters': 'standard',
            'network_type': 'combined',
            'load_population': True,
            'pop_location': 'initial_populations/saved_population_combined_GC_48.npy',
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
    'combined_tsp':
        {
            'config_name': 'test_evol_tsp',
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
    'combined_tsp_features':
        {
            'config_name': 'test_evol_tsp',
            'training_name': 'combined_feature_model',
            'learning_parameters': 'standard',
            'network_type': 'combined_features',
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
    'combined_tsp_features_onehot':
        {
            'config_name': 'test_evol_tsp',
            'training_name': 'combined_feature_onehot_model',
            'learning_parameters': 'standard',
            'network_type': 'combined_features_onehot',
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
    'combined_gc_features_onehot':
        {
            'config_name': 'test_evol_gc_onehot',
            'training_name': 'combined_feature_onehot_model',
            'learning_parameters': 'standard',
            'network_type': 'combined_features_onehot',
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
    'combined_gc_small':
        {
            'config_name': 'test_evol_gc_small',
            'training_name': 'combined_model',
            'learning_parameters': 'small',
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
    'combined_npp':
        {
            'config_name': 'test_evol_npp',
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
                'd': 0.2,
                'z': 0.1
            }
        },
    'combined_multiple':
        {
            'config_name': 'test_evol_multiple',
            'training_name': 'combined_multiple_model',
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
                'd': 0.2,
                'z': 0.05
            }
        },
    'combined_multiple_big':
        {
            'config_name': 'test_evol_multiple_big',
            'training_name': 'combined_multiple_model',
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
                'd': 0.2,
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
    'combined': CombinedModel,
    'combined_features': CombinedFeatureModel,
    'combined_features_onehot': CombinedOneHotFeatureModel,
    'combined_scale': CombinedScaleModel
}

fitness_function_generation_config = {
    'standard': construct_standard_fitness_function,
    'scale': construct_scale_fitness_function
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
    },
    'small': {
        'population': 100,
        'num_generations': 20,
        'keep_elitism': 5,
        'percent_of_parents_mating': 0.2
    },
    'long': {
        'population': 100,
        'num_generations': 1000,
        'keep_elitism': 5,
        'percent_of_parents_mating': 0.2
    }
}
