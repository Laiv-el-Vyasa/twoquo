from analysis_pipeline import AnalysisPipeline

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

if __name__ == "__main__":
    analysis_pipeline = AnalysisPipeline(scale_analysis_pipeline)
    analysis_pipeline.start_visualisation()
