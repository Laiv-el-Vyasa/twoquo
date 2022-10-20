from torch import torch, nn

from config import load_cfg
import pygad.torchga

cfg = load_cfg(cfg_id='test_evol')
qubo_size = cfg['pipeline']['problems']['qubo_size']
qubo_entries = (qubo_size * (qubo_size + 1)) / 2
population = 100


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(qubo_entries, qubo_entries)
        self.linear2 = nn.Linear(qubo_entries, qubo_entries)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return nn.ReLU(x)


#model = Network()
model = nn.Sequential(
    nn.Linear(qubo_entries, qubo_entries),
    nn.Linear(qubo_entries, qubo_entries),
    nn.ReLU()
)

torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=population)