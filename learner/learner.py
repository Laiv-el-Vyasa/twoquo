import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset_setup import DatabaseSetup, Data
from sklearn.model_selection import train_test_split

solver = 'qbsolv_simulated_annealing'
lower_bound = 0.95
batch_size = 4
problem_number = 3

db_setup = DatabaseSetup(3, solver)
X, Y = db_setup.get_data_for_simple_learning(lower_bound)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.33)

dataset = Data(x_train, y_train)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

#Model
input_dim = 3
hidden_dim = 10
output_dim = 1


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


network = Network()

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters(), lr=.01)

epochs = 20
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        x, y = data
        optimizer.zero_grad()
        outputs = network.forward(x)
        #if i == 0:
        #    print(x, y)
        #    print(outputs)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'{epoch + 1} loss: {running_loss}')


#dataloader_test = DataLoader(dataset, shuffle=True)
X_test = torch.from_numpy(x_test.astype(np.float32))
Y_test = torch.from_numpy(y_test.astype(np.float32))
print(f'Test: {x_test[0:5]}')
Y_test_result = network.forward(X_test)
print(f'Test result: {Y_test_result[0].detach().numpy()}')
result_error = loss_function(Y_test_result, Y_test)
print(f'Result error: {result_error}')

db_setup.visualize_results(network)

#print(db_setup.aggregate_saved_problem_data(3, solver))