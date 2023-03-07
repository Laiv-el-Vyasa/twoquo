import numpy as np

from config import load_cfg
from evolution_new.combined_evolution_training import get_data_from_training_config
from evolution_new.pygad_learning import PygadLearner
from learning_model import LearningModel


class TrainingAnalysis:
    def __init__(self, config_name: str, analysis_parameters: dict):
        self.model, learning_parameters, fitness_func = get_data_from_training_config(config_name)
        self.pygad_learner = PygadLearner(self.model, learning_parameters, fitness_func)
        self.training_name = learning_parameters['training_name']
        self.config = load_cfg(cfg_id=learning_parameters['config_name'])
        self.analysis_parameters = analysis_parameters
        if not self.model.load_best_model(self.training_name):
            self.pygad_learner.save_best_model()
            self.model.load_best_model(self.training_name)

    def get_simple_approx_baseline(self) -> list[list, list]:
        try:
            analysis_baseline = np.load(f'analysis_baseline/{self.training_name}.npk')
        except FileNotFoundError:
            analysis_baseline = self.get_analysis_baseline()
            np.save(f'analysis_baseline/{self.training_name}', analysis_baseline)
        return analysis_baseline

    