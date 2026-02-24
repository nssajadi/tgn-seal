import numpy as np
import torch
import torch.nn.functional as fn
from torch.nn import (
    ModuleList,
    Linear,
    Conv1d,
    MaxPool1d,
    Embedding,
    ReLU,
    Sequential,
    BatchNorm1d as BN,
)
from torch_geometric.nn import (
    MLP,
    GCNConv,
    GINConv,
    SAGEConv,
    global_sort_pool,
    global_mean_pool,
)


class GIN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden_channels,
        max_z,
        num_layers,
        dropout=0.5,
        jk=True,
        train_eps=False,
    ):
        super(GIN, self).__init__()

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk

        num_features += max_z

        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ),
            train_eps=train_eps,
        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ),
                    train_eps=train_eps,
                )
            )

        self.dropout = dropout
        if self.jk:
            self.lin1 = Linear(num_layers * hidden_channels, hidden_channels)
        else:
            self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x, z, edge_index, batch, edge_weight=None, node_id=None):
        device = self.conv1.nn[0].weight.device

        z = z.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)

        z_emb = fn.one_hot(z, self.max_z).to(torch.float).to(device)
        x = torch.cat([z_emb, x.to(torch.float)], 1)

        x = torch.cat([z_emb, x.to(torch.float)], 1)
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.jk:
            x = global_mean_pool(torch.cat(xs, dim=1), batch)
        else:
            x = global_mean_pool(xs[-1], batch)
        x = fn.relu(self.lin1(x))
        x = fn.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


class DGCNN(torch.nn.Module):
    def __init__(
        self, num_features, hidden_channels, max_z, num_layers, GNN=GCNConv, k=40
    ):
        super().__init__()
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        num_features += hidden_channels
        # num_features += max_z

        self.convs = ModuleList()
        self.convs.append(GNN(num_features, hidden_channels, add_self_loops=False))
        for i in range(0, num_layers - 1):
            self.convs.append(
                GNN(hidden_channels, hidden_channels, add_self_loops=False)
            )
        self.convs.append(GNN(hidden_channels, 1, add_self_loops=False))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        # self.mlp = MLP([dense_dim, 128, 1], dropout=0.1, batch_norm=False)
        self.mlp = MLP([dense_dim, 128, 1], batch_norm=False)

    def forward(self, x, z, edge_index, batch):
        device = self.z_embedding.weight.device
        z = z.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)

        z_emb = self.z_embedding(z)
        # z_emb = fn.one_hot(z, self.max_z).to(torch.float)
        x = torch.cat([z_emb, x.to(torch.float)], 1)

        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = self.conv1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv2(x).relu()
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        return self.mlp(x)


class SAGE(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden_channels,
        max_z,
        num_layers,
        use_feature=False,
        node_embedding=None,
        dropout=0.5,
    ):
        super(SAGE, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(SAGEConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, z, edge_index, batch, edge_weight=None, node_id=None):
        device = self.z_embedding.weight.device

        z = z.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        if x is not None:
            x = x.to(device)
        if node_id is not None:
            node_id = node_id.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = fn.relu(x)
            x = fn.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = x_src * x_dst
            x = fn.relu(self.lin1(x))
            x = fn.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden_channels,
        max_z,
        num_layers,
        use_feature=False,
        node_embedding=None,
        dropout=0.5,
    ):
        super(GCN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, z, edge_index, batch, edge_weight=None, node_id=None):
        device = self.z_embedding.weight.device

        z = z.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        if x is not None:
            x = x.to(device)
        if node_id is not None:
            node_id = node_id.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = fn.relu(x)
            x = fn.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = x_src * x_dst
            x = fn.relu(self.lin1(x))
            x = fn.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        else:  # sum pooling
            x = global_add_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


def get_link_pred_module(
    module_type, num_features, max_z=100, hidden_channels=32, num_layers=3
):
    if module_type == "dgcnn":
        return DGCNN(
            num_features,
            max_z=max_z,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )
    elif module_type == "gin":
        return GIN(
            num_features,
            max_z=max_z,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )
    elif module_type == "sage":
        return SAGE(
            num_features,
            max_z=max_z,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )
    elif module_type == "gcn":
        return GCN(
            num_features,
            max_z=max_z,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )
    else:
        return MergeLayer(num_features, num_features, num_features, 1)
