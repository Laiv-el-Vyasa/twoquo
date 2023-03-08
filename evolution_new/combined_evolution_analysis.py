from model_analysis import TrainingAnalysis

analysis_parameters = {
    'steps': 100,
    'analysis_name': 'combined_analysis_100_steps_NP',
    'boxplot': True,
    'sorted': True
}

if __name__ == "__main__":
    config_name = 'combined_npp'
    training_analysis = TrainingAnalysis(config_name, analysis_parameters)
    training_analysis.run_analysis()
