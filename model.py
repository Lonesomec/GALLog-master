from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, SAGEConv, GatedGraphConv, GATConv, \
    AttentionalAggregation, VGAE, GATv2Conv
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import negative_sampling


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        # for gossip
        self.conv1 = GCNConv(num_node_features, 512)
        self.conv2 = GCNConv(512, 256)
        self.fc = torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.Linear(128, 32),
                                      torch.nn.Linear(32, num_classes))

        # self.attention_pooling = AttentionalAggregation(self.fc)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        embs = global_max_pool(x, data.batch)
        x = self.fc(x)
        x = global_max_pool(x, data.batch)
        return x, embs


class GAT_NET(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads):
        super(GAT_NET, self).__init__()
        self.gat1 = GATv2Conv(features, hidden, heads, dropout=0.1)
        self.gat2 = GATv2Conv(hidden * heads, 256, 1, dropout=0.1)
        self.fc = torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.Linear(128, 32),
                                      torch.nn.Linear(32, classes))

    def forward(self, data):
        x = self.gat1(data.x, data.edge_index)
        x = F.leaky_relu(x)
        x = self.gat2(x, data.edge_index)
        embs = global_max_pool(x, data.batch)
        x = self.fc(x)
        x = global_max_pool(x, data.batch)
        return x, embs


class VariationalGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalGraphEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class VGAEModel(VGAE):
    def __init__(self, encoder, outs):
        super(VGAEModel, self).__init__(encoder)
        self.decoder = NodeFeatureDecoder(256, outs)

    def rec_loss_small(self, x, z, edge_index, criterion):
        # decoder计算的是每对节点之间的连边概率 而非特征
        pos_loss = criterion(x, self.decoder(z))
        return pos_loss


class NodeFeatureDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(NodeFeatureDecoder, self).__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return self.linear(z)
