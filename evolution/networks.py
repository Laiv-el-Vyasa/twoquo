from torch import torch, nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn


class GcnIdSimple(nn.Module):
    def __init__(self, qubo_size):
        super(GcnIdSimple, self).__init__()
        self.conv1 = geo_nn.GCNConv(qubo_size, int(qubo_size / 2), add_self_loops=False)
        self.conv2 = geo_nn.GCNConv(int(qubo_size / 2), qubo_size, add_self_loops=False)

    def forward(self, x, edge_index, edge_weights):
        x = self.conv1(x, edge_index, edge_weights)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weights)
        return F.relu(x)


class GcnIdStraight(nn.Module):
    def __init__(self, qubo_size):
        super(GcnIdStraight, self).__init__()
        self.conv1 = geo_nn.GCNConv(qubo_size, qubo_size, add_self_loops=False)
        self.conv2 = geo_nn.GCNConv(qubo_size, qubo_size, add_self_loops=False)

    def forward(self, x, edge_index, edge_weights):
        x = self.conv1(x, edge_index, edge_weights)
        x = self.conv2(x, edge_index, edge_weights)
        return F.relu(x)


class GcnDiag(nn.Module):
    def __init__(self, qubo_size, steps):
        super(GcnDiag, self).__init__()
        self.qubo_size = qubo_size
        self.steps = steps
        self.conv1 = geo_nn.GCNConv(1, int((1/5) * qubo_size), add_self_loops=True)
        self.conv2 = geo_nn.GCNConv(int((1/5) * qubo_size), int((2/5) * qubo_size), add_self_loops=True)
        self.conv3 = geo_nn.GCNConv(int((2/5) * qubo_size), int((3/5) * qubo_size), add_self_loops=True)
        self.conv4 = geo_nn.GCNConv(int((3/5) * qubo_size), int((4/5) * qubo_size), add_self_loops=True)
        self.conv5 = geo_nn.GCNConv(int((4/5) * qubo_size), qubo_size, add_self_loops=True)
        #self.conv_list = self.create_conv_list()

    def forward(self, x, edge_index, edge_weights):
        #for conv in self.conv_list:
        #    x = conv(x, edge_index, edge_weights)
        #    x = F.relu(x)
        x = self.conv1(x, edge_index, edge_weights)
        x = self.conv2(x, edge_index, edge_weights)
        x = self.conv3(x, edge_index, edge_weights)
        x = self.conv4(x, edge_index, edge_weights)
        x = self.conv5(x, edge_index, edge_weights)
        return F.relu(x)

    def create_conv_list(self):
        conv_list = []
        last_feature_count = 1
        for i in range(self.steps):
            next_feature_count = self.get_next_feature_count(i)
            conv_list.append(geo_nn.GCNConv(last_feature_count, next_feature_count , add_self_loops=True))
        return conv_list

    def get_next_feature_count(self, i):
        return int(((i + 1) / self.steps) * self.qubo_size)


