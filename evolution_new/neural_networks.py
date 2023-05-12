import numpy as np
from torch import torch, nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn
cuda = torch.device('cuda')


class Network(nn.Module):
    def __init__(self, qubo_size):
        super(Network, self).__init__()
        qubo_entries = int((qubo_size) * (qubo_size + 1) / 2)
        self.linear1 = nn.Linear(qubo_entries, qubo_entries)
        self.linear2 = nn.Linear(qubo_entries, qubo_entries)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return F.relu(x)


class AutoEnc(nn.Module):
    def __init__(self, qubo_size):
        super(AutoEnc, self).__init__()
        qubo_entries = int((qubo_size) * (qubo_size + 1) / 2)
        self.linear1 = nn.Linear(qubo_entries, int(qubo_entries / 2))
        self.linear2 = nn.Linear(int(qubo_entries/2), int(qubo_entries/4))
        self.linear3 = nn.Linear(int(qubo_entries/4), int(qubo_entries/2))
        self.linear4 = nn.Linear(int(qubo_entries/2), qubo_entries)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return F.relu(x)


class SqrtAutoEnc(nn.Module):
    def __init__(self, qubo_size):
        super(SqrtAutoEnc, self).__init__()
        qubo_entries = int(qubo_size * (qubo_size + 1) / 2)
        self.linear1 = nn.Linear(qubo_entries, int(np.sqrt(qubo_entries)))
        self.linear2 = nn.Linear(int(np.sqrt(qubo_entries)), int(np.sqrt(qubo_entries) / 2))
        self.linear3 = nn.Linear(int(np.sqrt(qubo_entries) / 2), int(np.sqrt(qubo_entries)))
        self.linear4 = nn.Linear(int(np.sqrt(qubo_entries)), qubo_entries)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return F.relu(x)


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


class GcnDeep(nn.Module):
    def __init__(self, qubo_size):
        super(GcnDeep, self).__init__()
        self.conv1 = geo_nn.GCNConv(qubo_size, int(qubo_size / 2), add_self_loops=True)
        self.conv2 = geo_nn.GCNConv(int(qubo_size / 2), int(qubo_size / 4), add_self_loops=True)
        self.conv3 = geo_nn.GCNConv(int(qubo_size / 4), int(qubo_size / 2), add_self_loops=True)
        self.conv4 = geo_nn.GCNConv(int(qubo_size / 2), qubo_size, add_self_loops=True)

    def forward(self, x, edge_index, edge_weights):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
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
        #print('X1: ', x)
        x = self.conv2(x, edge_index, edge_weights)
        #print('X2: ', x)
        x = self.conv3(x, edge_index, edge_weights)
        #print('X3: ', x)
        x = self.conv4(x, edge_index, edge_weights)
        #print('X4: ', x)
        x = self.conv5(x, edge_index, edge_weights)
        #print('X5: ', x)
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


class CombinedNodeFeatures(nn.Module):
    def __init__(self, node_features):
        super(CombinedNodeFeatures, self).__init__()
        self.conv1 = geo_nn.GCNConv(1, int(np.power(2, 1) * node_features / 8), add_self_loops=False, normalize=False)
        self.conv2 = geo_nn.GCNConv(int(np.power(2, 1) * node_features / 8),
                                    int(np.power(2, 2) * node_features / 8), add_self_loops=False, normalize=False)
        self.conv3 = geo_nn.GCNConv(int(np.power(2, 2) * node_features / 8), node_features,
                                    add_self_loops=False, normalize=False)

    def forward(self, x, edge_index, edge_weights):
        #print(x)
        x = self.conv1(x, edge_index, edge_weights)
        #print(x)
        #x = torch.sigmoid(x)
        #print(x)
        x = self.conv2(x, edge_index, edge_weights)
        #print(x)
        #x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weights)
        return torch.sigmoid(x)


class CombinedEdgeDecision(nn.Module):
    def __init__(self , node_features):
        super(CombinedEdgeDecision, self).__init__()
        self.lin1 = nn.Linear(node_features, int(np.power(2, 2) * node_features / 8))
        self.lin2 = nn.Linear(int(np.power(2, 2) * node_features / 8), int(np.power(2, 1) * node_features / 8))
        self.lin3 = nn.Linear(int(np.power(2, 1) * node_features / 8), 1)

    def forward(self, x):
        x = self.lin1(x)
        #x = F.relu(x)
        x = self.lin2(x)
        #x = torch.sigmoid(x)
        x = self.lin3(x)
        return F.relu(x)


class CombinedNodeFeaturesNonLin(nn.Module):
    def __init__(self, node_features):
        super(CombinedNodeFeaturesNonLin, self).__init__()
        self.conv1 = geo_nn.GCNConv(1, int(np.power(2, 1) * node_features / 8), add_self_loops=False, normalize=False)
        self.conv2 = geo_nn.GCNConv(int(np.power(2, 1) * node_features / 8),
                                    int(np.power(2, 2) * node_features / 8), add_self_loops=False, normalize=False)
        self.conv3 = geo_nn.GCNConv(int(np.power(2, 2) * node_features / 8), node_features,
                                    add_self_loops=False, normalize=False)

    def forward(self, x, edge_index, edge_weights):
        #print(x)
        x = self.conv1(x, edge_index, edge_weights)
        #print(x)
        x = torch.sigmoid(x)
        #print(x)
        x = self.conv2(x, edge_index, edge_weights)
        #print(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weights)
        return torch.sigmoid(x)


class CombinedEdgeDecisionNonLin(nn.Module):
    def __init__(self , node_features):
        super(CombinedEdgeDecisionNonLin, self).__init__()
        self.lin1 = nn.Linear(node_features, int(np.power(2, 2) * node_features / 8))
        self.lin2 = nn.Linear(int(np.power(2, 2) * node_features / 8), int(np.power(2, 1) * node_features / 8))
        self.lin3 = nn.Linear(int(np.power(2, 1) * node_features / 8), 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
        return F.relu(x)


class CombinedNodeFeaturesUwu(nn.Module):
    def __init__(self, node_features: int, normalized: bool):
        super(CombinedNodeFeaturesUwu, self).__init__()
        self.conv1 = geo_nn.GCNConv(1, int(np.power(2, 2) * node_features / 8),
                                    add_self_loops=False, normalize=normalized)
        self.conv2 = geo_nn.GCNConv(int(np.power(2, 2) * node_features / 8),
                                    int(np.power(2, 3) * node_features / 8),
                                    add_self_loops=False, normalize=normalized)
        self.conv3 = geo_nn.GCNConv(int(np.power(2, 3) * node_features / 8),
                                    int(np.power(2, 4) * node_features / 8),
                                    add_self_loops=False, normalize=normalized)
        self.conv4 = geo_nn.GCNConv(int(np.power(2, 4) * node_features / 8),
                                    int(np.power(2, 3) * node_features / 8),
                                    add_self_loops=False, normalize=normalized)
        self.conv5 = geo_nn.GCNConv(int(np.power(2, 3) * node_features / 8),
                                    int(np.power(2, 2) * node_features / 8),
                                    add_self_loops=False, normalize=normalized)

    def forward(self, x, edge_index, edge_weights):
        #print(x)
        x = self.conv1(x, edge_index, edge_weights)
        #print(x)
        x = torch.sigmoid(x)
        #print(x)
        x = self.conv2(x, edge_index, edge_weights)
        #print(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weights)
        x = torch.sigmoid(x)
        x = self.conv4(x, edge_index, edge_weights)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weights)
        return torch.sigmoid(x)


class CombinedEdgeDecisionUwu(nn.Module):
    def __init__(self, node_features):
        super(CombinedEdgeDecisionUwu, self).__init__()
        self.lin1 = nn.Linear(int(np.power(2, 2) * node_features / 8), int(np.power(2, 3) * node_features / 8))
        self.lin2 = nn.Linear(int(np.power(2, 3) * node_features / 8), int(np.power(2, 4) * node_features / 8))
        self.lin3 = nn.Linear(int(np.power(2, 4) * node_features / 8), int(np.power(2, 3) * node_features / 8))
        self.lin4 = nn.Linear(int(np.power(2, 3) * node_features / 8), int(np.power(2, 2) * node_features / 8))
        self.lin5 = nn.Linear(int(np.power(2, 2) * node_features / 8), 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        x = torch.sigmoid(x)
        x = self.lin5(x)
        return F.relu(x)


class CombinedScaleEdgeDecisionUwu(nn.Module):
    def __init__(self, node_features):
        super(CombinedScaleEdgeDecisionUwu, self).__init__()
        self.lin1 = nn.Linear(int(np.power(2, 2) * node_features / 8 + 1), int(np.power(2, 3) * node_features / 8))
        self.lin2 = nn.Linear(int(np.power(2, 3) * node_features / 8), int(np.power(2, 4) * node_features / 8))
        self.lin3 = nn.Linear(int(np.power(2, 4) * node_features / 8), int(np.power(2, 3) * node_features / 8))
        self.lin4 = nn.Linear(int(np.power(2, 3) * node_features / 8), int(np.power(2, 2) * node_features / 8))
        self.lin5 = nn.Linear(int(np.power(2, 2) * node_features / 8), 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        x = torch.sigmoid(x)
        x = self.lin5(x)
        return F.relu(x)