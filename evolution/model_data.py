from evolution.networks import CombinedNodeFeatures, CombinedEdgeDecision, CombinedNodeFeaturesNonLin, \
    CombinedEdgeDecisionNonLin, CombinedNodeFeaturesUwu, CombinedEdgeDecisionUwu, GcnIdSimple, GcnIdStraight, GcnDiag, \
    GcnDeep, Network, AutoEnc, SqrtAutoEnc
from evolution_util import qubo_size

node_feature_number = 8

evaluation_models = {
    'combined_evolution_MC_8_1_05_10_01_005':
        {'name': 'combined model, 8',
         'fitness_params': (1, .5, 10, .1),
         'min_approx': 0.05,
         'independence': True,
         'model_name': '',
         'evolution_type': 'combined',
         'display': False
         },
    'combined_evolution_MC_24_1_05_10_01_005':
        {'name': 'combined model, 24',
         'fitness_params': (1, .5, 10, .1),
         'min_approx': 0.05,
         'independence': True,
         'model_name': '',
         'evolution_type': 'combined',
         'display': False
         },
    'combined_evolution_MC_24_nonlin_1_05_10_01_005':
        {'name': 'combined model, 24, non-linear',
         'fitness_params': (1, .5, 10, .1),
         'min_approx': 0.05,
         'independence': True,
         'model_name': '_nonlin',
         'evolution_type': 'combined',
         'display': False
         },
    'combined_evolution_MC_24_uwu_1_05_10_01_005':
        {'name': 'combined model, MC, 24, compression',
         'fitness_params': (1, .5, 10, .1),
         'min_approx': 0.05,
         'independence': True,
         'model_name': '_uwu',
         'evolution_type': 'combined',
         'display': True
         },
    'combined_evolution_MC_24_uwu_1_1_10_02_01_ext_1_05_10_01_005':
        {'name': 'combined model, 24, compression, increase approx, extended',
         'fitness_params': (1, 1, 10, .2),
         'min_approx': 0.1,
         'independence': True,
         'model_name': '_uwu',
         'evolution_type': 'combined',
         'display': False
         },
    'combined_evolution_MC_24_uwu_1_1_10_02_01':
        {'name': 'combined model, 24, compression, increase approx, 100gen',
         'fitness_params': (1, 1, 10, .2),
         'min_approx': 0.1,
         'independence': True,
         'model_name': '_uwu',
         'evolution_type': 'combined',
         'display': False
         },
    'combined_evolution_MC_24_uwu_1_1_10_02_01_200gen':
        {'name': 'combined model, 24, compression, increase approx, 200gen',
         'fitness_params': (1, 1, 10, .2),
         'min_approx': 0.1,
         'independence': True,
         'model_name': '_uwu',
         'evolution_type': 'combined',
         'display': False
         },
    'combined_evolution_MC_48_uwu_1_05_10_01_005': {
        'name': 'combined model, 48, compression',
        'fitness_params': (1, .5, 10, .1),
        'min_approx': 0.05,
        'independence': True,
        'model_name': '_uwu',
        'evolution_type': 'combined',
        'display': False
    },
    'combined_evolution_MC_48_uwu_1_05_10_01_005_ext_24_1_05_10_01_005': {
        'name': 'combined model, 48, compression, extension of 24 model',
        'fitness_params': (1, .5, 10, .1),
        'min_approx': 0.05,
        'independence': True,
        'model_name': '_uwu',
        'evolution_type': 'combined',
        'display': False
    },
    'combined_evolution_NP_24_uwu_1_05_10_02_01': {
        'name': 'combined model, 24, compression',
        'fitness_params': (1, .5, 10, .2),
        'min_approx': 0.1,
        'independence': True,
        'model_name': '_uwu',
        'evolution_type': 'combined',
        'display': True
    },
    'gcn_evolution_MC_8_1_05_10_1': {
        'name': 'gcn model, 24, id-matrix',
        'fitness_params': (1, .5, 10, 1),
        'min_approx': 0,
        'model_name': '',
        'independence': False,
        'evolution_type': 'gcn',
        'display': False
    },
    'gcn_evolution_MC_24_1_05_10_01': {
        'name': 'gcn model, 24, id-matrix, correct solutions',
        'fitness_params': (1, .5, 10, .1),
        'min_approx': 0,
        'model_name': '',
        'independence': False,
        'evolution_type': 'gcn',
        'display': False
    },
    'gcn_evolution_MC_24_1_1_1_1': {
        'name': 'gcn model, 24, id-matrix, balanced',
        'fitness_params': (1, 1, 1, 1),
        'min_approx': 0,
        'model_name': '',
        'independence': False,
        'evolution_type': 'gcn',
        'display': False
    },
    'gcn_evolution_MC_24_1_1_1_1_005': {
        'name': 'gcn model, 24, id-matrix, balanced min .05',
        'fitness_params': (1, 1, 1, 1),
        'min_approx': .05,
        'model_name': '',
        'independence': False,
        'evolution_type': 'gcn',
        'display': False
    },
    'gcn_evolution_MC_24_straight_1_2_1_1_005': {
        'name': 'gcn straight model, 24, id-matrix',
        'fitness_params': (1, 2, 1, 1),
        'min_approx': .05,
        'model_name': '_straight',
        'independence': False,
        'evolution_type': 'gcn',
        'display': False
    },
    'gcn_evolution_MC_24_diag_1_1_1_1_005': {
        'name': 'gcn single inputs, 24, qubo-diagonal',
        'fitness_params': (1, 1, 1, 1),
        'min_approx': 0.05,
        'model_name': '_diag',
        'independence': False,
        'evolution_type': 'gcn',
        'display': False
    },
    'simple_evolution_NP_8_1_1_0_05':
        {'name': 'simple model, 8, target quality & .5 approxed entries',
         'fitness_params': (1, 1, 0, .5),
         'min_approx': 0,
         'model_name': '',
         'independence': False,
         'evolution_type': 'simple',
         'display': False
         },
    'simple_evolution_NP_8_1_1_0_08':
        {'name': 'simple model, 8, target quality & .8 approxed entries',
         'fitness_params': (1, 1, 0, .8),
         'min_approx': 0,
         'model_name': '',
         'independence': False,
         'evolution_type': 'simple',
         'display': False
         },
    'simple_evolution_NP_8_1_05_10_1':
        {'name': 'simple model, 8, target correct_solutions',
         'fitness_params': (1, .5, 10, 1),
         'min_approx': 0,
         'model_name': '',
         'independence': False,
         'evolution_type': 'simple',
         'display': False
         },
    'simple_evolution_NP_24_1_05_10_01_20p': {
        'name': 'simple model, 24, target correct solutions',
        'fitness_params': (1, .5, 10, .1),
        'min_approx': 0,
        'model_name': '',
        'independence': False,
        'evolution_type': 'simple',
        'display': True
    },
    'simple_evolution_NP_24_1_1_0_09_20p': {
        'name': 'simple model, 24, target quality & .9 approxed entries',
        'fitness_params': (1, 1, 0, .9),
        'min_approx': 0,
        'model_name': '',
        'independence': False,
        'evolution_type': 'simple',
        'display': False
    },
    'simple_evolution_MC_8_1_05_10_1_MC': {
        'name': 'simple model, 8, target correct solutions',
        'fitness_params': (1, .5, 10, 1),
        'min_approx': 0,
        'model_name': '',
        'independence': False,
        'evolution_type': 'simple',
        'display': False
    },
    'simple_evolution_MC_24_1_05_10_1': {
        'name': 'simple model, 24, target increased approx',
        'min_approx': 0,
        'fitness_params': (1, .5, 10, 1),
        'model_name': '',
        'independence': False,
        'evolution_type': 'simple',
        'display': False
    },
    'simple_evolution_MC_24_1_05_10_01': {
        'name': 'simple model, 24',
        'fitness_params': (1, .5, 10, .1),
        'min_approx': 0,
        'model_name': '',
        'independence': False,
        'evolution_type': 'simple',
        'display': False
    },
    'simple_evolution_MC_24_autoenc_1_05_10_01': {
        'name': 'simple model, 24, autoencoder',
        'fitness_params': (1, .5, 10, .1),
        'min_approx': 0,
        'model_name': '_autoenc',
        'independence': False,
        'evolution_type': 'simple',
        'display': False
    },
    'simple_evolution_MC_24_sqrt_autoenc_1_05_10_01': {
        'name': 'simple model, 24, sqrt autoencoder',
        'fitness_params': (1, .5, 10, .1),
        'min_approx': 0,
        'model_name': '_sqrt_autoenc',
        'independence': False,
        'evolution_type': 'simple',
        'display': False
    }
}

model_dict = {
    'model': [
        CombinedNodeFeatures(node_feature_number),
        CombinedEdgeDecision(node_feature_number)
    ],
    'model_nonlin': [
        CombinedNodeFeaturesNonLin(node_feature_number),
        CombinedEdgeDecisionNonLin(node_feature_number)
    ],
    'model_uwu': [
        CombinedNodeFeaturesUwu(node_feature_number),
        CombinedEdgeDecisionUwu(node_feature_number)
    ],
    'combined': {
        'model': [
            CombinedNodeFeatures(node_feature_number),
            CombinedEdgeDecision(node_feature_number)
        ],
        'model_nonlin': [
            CombinedNodeFeaturesNonLin(node_feature_number),
            CombinedEdgeDecisionNonLin(node_feature_number)
        ],
        'model_uwu': [
            CombinedNodeFeaturesUwu(node_feature_number),
            CombinedEdgeDecisionUwu(node_feature_number)
        ]
    },
    'gcn': {
        'model': GcnIdSimple(qubo_size),
        'model_straight': GcnIdStraight(qubo_size),
        'model_diag': GcnDiag(qubo_size, 5),
        'model_deep': GcnDeep(qubo_size)
    },
    'simple': {
        'model': Network(qubo_size),
        'model_autoenc': AutoEnc(qubo_size),
        'model_sqrt_autoenc': SqrtAutoEnc(qubo_size)
    }
}
