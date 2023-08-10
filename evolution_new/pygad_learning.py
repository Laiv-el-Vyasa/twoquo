import numpy as np
import pygad
from time import time as ttime
from config import load_cfg
from learning_model import LearningModel
from typing import Callable

from evolution_utils import delete_data, get_file_name
from new_visualisation import plot_average_fitness


# Class to manage the training process of the modes using a GA
class PygadLearner:
    def __init__(self,
                 model: LearningModel,
                 parameters: dict,
                 fitness_function: Callable[[dict, dict], float]
                 ):
        delete_data()
        self.model = model
        self.learning_parameters = parameters
        self.config = load_cfg(cfg_id=parameters['config_name'])
        self.training_name = parameters['training_name']
        self.best_fitness = 0
        self.avg_fitness_list = np.array([])
        self.avg_fitness_generation = 0
        self.avg_fitness_generation_count = 0
        self.avg_duration = []
        self.get_fitness_value = fitness_function
        self.ga_instance = self.initialize_ga_instance()

    # Loading a saved GA-Instance or creating a new one
    def initialize_ga_instance(self) -> pygad.GA:
        try:
            ga_instance = pygad.load(f'pygad_trainings/{self.training_name}')
            ga_instance.on_generation = self.callback_generation
            ga_instance.fitness_func = self.get_fitness_function()
            self.avg_fitness_list = np.load(f'pygad_trainings/avg_fitness_lists/{self.training_name}_avg_fitness.npy')
        except FileNotFoundError:
            num_parents_mating = int(self.learning_parameters['population'] *
                                     self.learning_parameters['percent_of_parents_mating'])
            initial_population = self.model.get_initial_population(self.learning_parameters['population'])
            if 'load_population' in self.learning_parameters:
                try:
                    initial_population = np.load(self.learning_parameters['pop_location'])
                    print('Initial population loaded')
                except FileNotFoundError:
                    pass
            ga_instance = pygad.GA(num_generations=self.learning_parameters['num_generations'],
                                   # parent_selection_type='rws',
                                   keep_elitism=self.learning_parameters['keep_elitism'],
                                   # crossover_type='scattered',
                                   num_parents_mating=num_parents_mating,
                                   initial_population=initial_population,
                                   fitness_func=self.get_fitness_function(),
                                   on_generation=self.callback_generation)
            print('New training created')
        return ga_instance

    def learn_model(self):
        self.ga_instance.run()
        self.ga_instance.save(f'pygad_trainings/{self.training_name}')
        self.ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness")
        plot_average_fitness(self.avg_fitness_list)
        self.save_best_model()
        np.save(f'pygad_trainings/avg_fitness_lists/{self.training_name}_avg_fitness', self.avg_fitness_list)

    def get_fitness_function(self) -> Callable[[list, int], float]:
        def fitness_function(solution, solution_idx):
            time = ttime()
            self.model.set_model_weights_from_pygad(solution)
            problem_dict = self.model.get_approximation(self.model.get_training_dataset(self.config))
            fitness = self.get_fitness_value(problem_dict, self.config)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
            self.set_avg_generation_fitness(fitness, solution_idx)
            print(f'Solution {solution_idx}: {fitness}')
            self.avg_duration.append(ttime() - time)
            return fitness

        return fitness_function

    # Tasks between generations
    def callback_generation(self, ga_instance: pygad.GA):
        print("Generation   = {generation}".format(generation=ga_instance.generations_completed))
        # print("Fitness      = {fitness}".format(fitness=ga_instance.best_solution()[1]))
        print("Fitness      = {fitness}".format(fitness=self.best_fitness))
        print('Avg fit generation: ', self.avg_fitness_generation)
        self.avg_fitness_list = np.append(self.avg_fitness_list, self.avg_fitness_generation)
        print("Avg. Fitness = {fitness}".format(fitness=self.avg_fitness_generation))
        self.avg_fitness_generation = 0
        self.avg_fitness_generation_count = 0
        print("Avg. Runtime = {time}".format(time=np.mean(self.avg_duration)))
        self.avg_duration = []

    def set_avg_generation_fitness(self, fitness: float, solution_id: int):
        self.avg_fitness_generation_count += 1
        self.avg_fitness_generation = (self.avg_fitness_generation * (self.avg_fitness_generation_count - 1) +
                                       fitness) / self.avg_fitness_generation_count

    def save_best_model(self):
        self.model.save_best_model(self.ga_instance.best_solution()[0], self.training_name)
