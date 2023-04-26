from model_analysis import TrainingAnalysis

analysis_parameters = {
    'steps': 100,
    'analysis_name': 'combined_feature_analysis',
    'boxplot': True,
    'sorted': True,
    'show_qubo_mask': 5,
    'size_analysis': True
}

if __name__ == "__main__":
    config_name = 'combined_tsp_features_onehot'
    training_analysis = TrainingAnalysis(config_name, analysis_parameters)
    training_analysis.run_analysis()
