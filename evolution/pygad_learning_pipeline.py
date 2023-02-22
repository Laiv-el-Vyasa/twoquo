import pygad

learning_parameters = {
    'config_name': 'test_evol_m3sat',
    'population': 100,
    'num_generations': 50,
    'percent_of_parents_mating': 0.2
}


def fitness_function():
    pass


def callback_generation():
    pass


fitness_function = fitness_function


def learn_model(model_name, initial_population):

    ga_instance = pygad.GA(num_generations=learning_parameters['num_generations'],
                           # parent_selection_type='rws',
                           keep_elitism=5,
                           # crossover_type='scattered',
                           num_parents_mating=learning_parameters['population'] * learning_parameters['percent_of_parents_mating'],
                           initial_population=initial_population,
                           fitness_func=fitness_function,
                           on_generation=callback_generation)
    ga_instance.run()
    ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness")
    #plot_average_fitness(avg_fitness_list)
    ga_instance.save(model_name)
