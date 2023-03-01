import pygad

from config import load_cfg
from learning_model import LearningModel
from typing import Callable

from evolution_utils import get_training_dataset
from visualisation import plot_average_fitness


class PygadLearner:
    def __init__(self,
                 model: LearningModel,
                 parameters: dict,
                 fitness_function: Callable[[list, list, list, dict], float]
                 ):
        self.model = model
        self.training_name = parameters['training_name']
        self.learning_parameters = parameters
        self.config = load_cfg(parameters['config_name'])
        self.get_fitness_value = fitness_function
        self.best_fitness = 0
        self.avg_fitness_list = []
        self.avg_fitness_generation = 0
        self.ga_instance = self.initialize_ga_instance()

    def initialize_ga_instance(self) -> pygad.GA:
        try:
            ga_instance = pygad.load(f'pygad_trainings/{self.training_name}')
        except FileNotFoundError:
            num_parents_mating = self.learning_parameters['population'] * \
                                 self.learning_parameters['percent_of_parents_mating']
            ga_instance = pygad.GA(num_generations=self.learning_parameters['num_generations'],
                                   # parent_selection_type='rws',
                                   keep_elitism=self.learning_parameters['keep_elitism'],
                                   # crossover_type='scattered',
                                   num_parents_mating=num_parents_mating,
                                   initial_population=self.model.get_initial_population(
                                       self.learning_parameters['population']
                                   ),
                                   fitness_func=self.get_fitness_function(),
                                   on_generation=self.callback_generation)
        return ga_instance

    def learn_model(self):
        self.ga_instance.run()
        self.ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness")
        plot_average_fitness(self.avg_fitness_list)
        self.ga_instance.save(f'pygad_trainings/{self.training_name}')
        self.model.save_best_model(self.ga_instance.best_solution()[0], self.training_name)

    def get_fitness_function(self) -> Callable[[list, int], float]:
        def fitness_function(solution, solution_idx):
            self.model.set_model_weights_from_pygad(solution)
            problem_dict = get_training_dataset(self.config)
            qubo_list = problem_dict['qubo_list']
            problem_list = problem_dict['problem_list']
            solution_list = problem_dict['solution_list]']
            approxed_qubo_list = self.model.get_approximation(qubo_list, problem_list)
            fitness = self.get_fitness_value(qubo_list, approxed_qubo_list, solution_list, self.config)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
            self.set_avg_generation_fitness(fitness, solution_idx)
            print(f'Solution {solution_idx}: {fitness}')
            return fitness

        return fitness_function

    def callback_generation(self):
        print("Generation   = {generation}".format(generation=self.ga_instance.generations_completed))
        # print("Fitness      = {fitness}".format(fitness=ga_instance.best_solution()[1]))
        print("Fitness      = {fitness}".format(fitness=self.best_fitness))
        # print(avg_fitness_generation)
        self.avg_fitness_list.append(self.avg_fitness_generation)
        print("Avg. Fitness = {fitness}".format(fitness=self.avg_fitness_generation))
        self.avg_fitness_generation = 0

    def set_avg_generation_fitness(self, fitness: float, solution_id: int):
        self.avg_fitness_generation = (self.avg_fitness_generation * (solution_id - 1) + fitness) / solution_id
