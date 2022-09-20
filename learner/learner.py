import torch
from dataset_setup import DatabaseSetup, Data
from sklearn.model_selection import train_test_split

solver = 'qbsolv_simulated_annealing'
lower_bound = 0.95

db_setup = DatabaseSetup()
X, Y = db_setup.get_data_for_simple_learning(3, solver, lower_bound)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.33)

dataset = Data(x_train, y_train)



print(dataset[0:23])

