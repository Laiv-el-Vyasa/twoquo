analysis_parameters = {
    'steps': 100,
    'analysis_name': 'combined_analysis',
    'boxplot': True,
    'sorted': True,
    'show_qubo_mask': 0,
    'size_analysis': False,
    'compare_different_approaches': True,
    'solver': 'qbsolv_simulated_annealing'
}

analysis_parameters_features_onehot = {
    'steps': 100,
    'analysis_name': 'combined_feature_onehot_analysis',
    'boxplot': True,
    'sorted': True,
    'show_qubo_mask': 0,
    'size_analysis': False,
    'compare_different_approaches': True,
    'solver': 'qbsolv_simulated_annealing'
}

scale_analysis_parameters = {
    'steps': 100,
    'analysis_name': 'combined_analysis',
    'boxplot': True,
    'sorted': True,
    'show_qubo_mask': 0,
    'size_analysis': False,
    'compare_different_approaches': True,
    'solver': 'qbsolv_simulated_annealing',
    'scale_list': [0.1, 0.2, 0.5]
}

scale_analysis_pipeline = {
    'models': {
        'combined_ec_scale': {
            'analysis_parameters': scale_analysis_parameters,
            'analysis_name': 'combined_analysis',
            'configs': [
                'standard', 'standard', 'standard'
            ]
        }
    },
    'analysis': [
        {
            'type': 'baseline_correct_mean',
            'compare': False,
            'models': {
                'combined_ec_scale': {
                    'model_name': 'Combined model trained on EC, 192',
                    'configs': [
                        0, 1, 2
                    ],
                    'colors': [
                        "black", "blue", "purple"
                    ],
                    'baseline_colors': [
                        "black", "black", "black"
                    ]
                }
            }
        }
    ]
}

analysis_pipeline = {
    'models': {
        'combined_ec': {
            'analysis_parameters': analysis_parameters,
            'analysis_name': 'combined_analysis',
            'configs': [
                'standard'
            ]
        }
    },
    'analysis': [
        {
            'type': 'baseline_correct_mean',
            'compare': True,
            'models': {
                'combined_ec': {
                    'model_name': 'Combined model trained on EC, 192',
                    'configs': [
                        0
                    ],
                    'colors': [
                        "black"
                    ],
                    'baseline_colors': [
                        "black"
                    ]
                }
            }
        },
        {
            'type': 'baseline_correct_incorrect',
            'model': 'combined_ec',
            'config': 0,
            'colors': ('green', 'black'),
            'baseline_color': 'grey'
        },
        {
            'type': 'relative_quality_with_mean',
            'model': 'combined_ec',
            'config': 0,
            'colors': ('blue', 'slateblue'),
            'baseline_colors': ('black', 'grey')
        },
        {
            'type': 'boxplot_one',
            'model': 'combined_ec',
            'config': 0
        }
    ]
}

analysis_pipeline_tsp = {
    'models': {
        'combined_tsp_features_onehot': {
            'analysis_parameters': analysis_parameters_features_onehot,
            'analysis_name': 'combined_feature_onehot_analysis',
            'configs': [
                'standard'
            ]
        }
    },
    'analysis': [
        {
            'type': 'baseline_correct_mean',
            'compare': False,
            'models': {
                'combined_tsp_features_onehot': {
                    'model_name': 'Combined feature onehot model\ntrained on TSP, 121',
                    'configs': [
                        0
                    ],
                    'colors': [
                        "black"
                    ],
                    'baseline_colors': [
                        "black"
                    ]
                }
            }
        },
        {
            'type': 'boxplot_one',
            'model': 'combined_tsp_features_onehot',
            'config': 0
        }
    ]
}


analysis_pipeline_sgi = {
    'models': {
        'combined_sgi': {
            'analysis_parameters': analysis_parameters,
            'analysis_name': 'combined_analysis',
            'configs': [
                'standard'
            ]
        }
    },
    'analysis': [
        {
            'type': 'baseline_correct_mean',
            'compare': False,
            'models': {
                'combined_sgi': {
                    'model_name': 'Combined model\ntrained on SGI, 96',
                    'configs': [
                        0
                    ],
                    'colors': [
                        "black"
                    ],
                    'baseline_colors': [
                        "black"
                    ]
                }
            }
        },
        {
            'type': 'boxplot_one',
            'model': 'combined_sgi',
            'config': 0
        }
    ]
}

analysis_pipeline_2 = {
    'models': {
        'combined_mc': {
            'analysis_parameters': analysis_parameters,
            'analysis_name': 'combined_analysis',
            'configs': [
                'standard'
            ]
        },
        'combined_gc': {
            'analysis_parameters': analysis_parameters,
            'analysis_name': 'combined_analysis',
            'configs': [
                'standard'
            ]
        },
        'combined_m3sat': {
            'analysis_parameters': analysis_parameters,
            'analysis_name': 'combined_analysis',
            'configs': [
                'standard'
            ]
        },
        'combined_npp': {
            'analysis_parameters': analysis_parameters,
            'analysis_name': 'combined_analysis',
            'configs': [
                'standard'
            ]
        },
        'combined_ec': {
            'analysis_parameters': analysis_parameters,
            'analysis_name': 'combined_analysis',
            'configs': [
                'standard'
            ]
        },
    },
    'analysis': [
        {
            'type': 'baseline_correct_mean',
            'compare': True,
            'models': {
                'combined_m3sat': {
                    'model_name': 'Combined model trained on M3SAT, 128',
                    'configs': [
                        0
                    ],
                    'colors': [
                        "black"
                    ],
                    'baseline_colors': [
                        "black"
                    ]
                }
            }
        },
        {
            'type': 'baseline_correct_mean',
            'compare': True,
            'models': {
                'combined_gc': {
                    'model_name': 'Combined model trained on GC, 128',
                    'configs': [
                        0
                    ],
                    'colors': [
                        "darkviolet"
                    ],
                    'baseline_colors': [
                        "black"
                    ]
                }
            }
        },
        {
            'type': 'boxplot_one',
            'model': 'combined_m3sat',
            'config': 0
        },
        {
            'type': 'boxplot_one',
            'model': 'combined_gc',
            'config': 0
        },
        {
            'type': 'boxplot_multiple',
            'models': {
                'combined_m3sat': {
                    'model_name': 'Combined model trained on M3SAT, 128',
                    'configs': [
                        0
                    ]
                },
                'combined_mc': {
                    'model_name': 'Combined model trained on MC, 128',
                    'configs': [
                        0
                    ]
                },
                'combined_ec': {
                    'model_name': 'Combined model trained on EC, 196',
                    'configs': [
                        0
                    ]
                },
                'combined_npp': {
                    'model_name': 'Combined model trained on NPP, 64',
                    'configs': [
                        0
                    ]
                }
            }
        }
    ]
}


analysis_pipeline_multiple = {
    'models': {
        'combined_multiple': {
            'analysis_parameters': analysis_parameters,
            'analysis_name': 'combined_analysis',
            'configs': [
                'test_evol_ec', 'test_evol_mc', 'test_evol_npp', 'test_evol_m3sat'
            ]
        }
    },
    'analysis': [
        {
            'type': 'baseline_correct_mean',
            'compare': False,
            'models': {
                'combined_multiple': {
                    'model_name': 'Combined model\ntrained on EC, MC, NPP, M3SAT 128',
                    'configs': [
                        0, 1, 2, 3
                    ],
                    'colors': [
                        'black', "violet", 'blue', 'teal'
                    ],
                    'baseline_colors': [
                        'black', "violet", 'blue', 'teal'
                    ]
                }
            }
        },
        {
            'type': 'boxplot_multiple',
            'models': {
                'combined_multiple': {
                    'model_name': 'Combined model\ntrained on EC, MC, NPP, M3SAT 128',
                    'configs': [
                        0, 1, 2, 3
                    ]
                }
            }
        }
    ]
}
