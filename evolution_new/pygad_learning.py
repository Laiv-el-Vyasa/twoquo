import pygad

from config import load_cfg
from learning_model import LearningModel

learning_parameters = {
    'config_name': 'test_evol_m3sat',
    'population': 100,
    'num_generations': 50,
    'percent_of_parents_mating': 0.2
}


class PygadLearner:
    def __init__(self,
                 model: LearningModel,
                 parameters: dict,
                 fitness_function: callable,
                 generation_callback: callable
                 ):
        self.model = model
        self.learning_parameters = parameters
        self.config = load_cfg(parameters['config_name'])
        self.fitness_function = fitness_function
        self.callback_generation = generation_callback

    def learn_model(self, model_name):
        num_parents_mating = self.learning_parameters['population'] * learning_parameters['percent_of_parents_mating']
        ga_instance = pygad.GA(num_generations=self.learning_parameters['num_generations'],
                               # parent_selection_type='rws',
                               keep_elitism=5,
                               # crossover_type='scattered',
                               num_parents_mating=num_parents_mating,
                               initial_population=self.model.get_initial_population(),
                               fitness_func=self.fitness_function,
                               on_generation=self.callback_generation)
        ga_instance.run()
        ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness")
        # plot_average_fitness(avg_fitness_list)
        ga_instance.save(model_name)

    def get_fitness_function(self):
        def fitness_function():
            problem_dict = self.model.get_training_dataset(self.config)
        return fitness_function
