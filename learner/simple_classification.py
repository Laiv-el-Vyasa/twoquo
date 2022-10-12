import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from config import load_cfg
from dataset_setup import DatabaseSetup, Data
from sklearn.model_selection import train_test_split

solver = 'qbsolv_simulated_annealing'
min_solution_quality = 0.95
learning_rate = .01
epochs = 200
batch_size = 4
cfg = load_cfg(cfg_id='test_small')
problem_number = len(cfg['pipeline']['problems']['problems'])

db_setup = DatabaseSetup(cfg, problem_number, solver, min_solution_quality)
X, Y, approx_steps = db_setup.get_data_for_simple_classification()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.33)

dataset = Data(x_train, y_train, classification=True)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

print(dataset[0])

#Model
input_dim = problem_number
hidden_dim = problem_number + approx_steps
output_dim = approx_steps


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


network = Network()

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)


for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        x, y = data
        optimizer.zero_grad()
        outputs = network.forward(x)
        log_softmax = nn.LogSoftmax(dim=1)
        #if i == 1 and epoch == 19:
        #    print(x, y)
        #    print(softmax(outputs))
        #x = softmax(x)
        loss = loss_function(log_softmax(outputs), y.type(torch.LongTensor))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'{epoch + 1} loss: {running_loss}')


#dataloader_test = DataLoader(dataset, shuffle=True)
X_test = torch.from_numpy(x_test.astype(np.float32))
Y_test = torch.from_numpy(y_test.astype(np.float32))
print(f'Test: {x_test[:4]}')
softmax = nn.Softmax(dim=1)
log_softmax = nn.LogSoftmax(dim=1)
Y_test_result = softmax(network.forward(X_test))
print(f'Test result: {Y_test_result[0].detach().numpy()}')
result_error = loss_function(log_softmax(Y_test_result), Y_test.type(torch.LongTensor))
print(f'Result error: {result_error}')

db_setup.visualize_results(network, 'classification')

#print(db_setup.aggregate_saved_problem_data(3, solver))