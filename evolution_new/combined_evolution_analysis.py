from model_analysis import TrainingAnalysis

analysis_parameters = {
    'steps': 100,
    'analysis_name': 'combined_analysis',
    'boxplot': True,
    'sorted': True,
    'show_qubo_mask': 0,
    'size_analysis': False,
    'compare_different_approaches': True
}

if __name__ == "__main__":
    config_name = 'combined_ec'
    training_analysis = TrainingAnalysis(config_name, analysis_parameters)
    training_analysis.run_analysis()
