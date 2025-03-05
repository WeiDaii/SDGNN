import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import Linear, ReLU, Dropout, LogSoftmax, Module
from scipy import sparse as sp
import math
import torch.nn.functional as F
import scipy
import scipy.sparse




class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.matmul(adj, out)
        return F.relu(out)
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = torch.matmul(adj, x)
        out = self.linear(support)
        return F.relu(out)
class SDGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_edge_types, num_nodes, num_steps, output_dim, dropout_rate=0.0, num_heads=1, One_part=True, Two_part=True, use_gcn=True, use_gnn=False,use_isattention=True, use_SiameseNet=True):
        super(SDGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.num_nodes = num_nodes
        self.num_steps = num_steps
        self.num_heads=num_heads
        self.use_isattention = use_isattention
        self.use_SiameseNet = use_SiameseNet
        self.SiameseNet = AT_SiameseNet(output_dim, 64, 2, num_heads=self.num_heads, use_isattention=self.use_isattention)
        self.No_Siamese_Linear = nn.Linear(2*num_nodes, 2)
        self.No_Siamese_LogSoftMax = nn.LogSoftmax(dim=1)
        self.W = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_edge_types)])
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.One_part_Linear_Softmax = nn.LogSoftmax(dim=1)
        self.One_part = One_part
        self.Two_part = Two_part
        self.use_gcn = use_gcn
        self.use_gnn = use_gnn
        self.gcn_layer = GCNLayer(input_dim, hidden_dim)
        self.gnn_layer = GNNLayer(input_dim, hidden_dim)
        self.GNN_part_Linear = nn.Linear(hidden_dim, num_nodes)

    def forward(self, features, adjacency_matrices, edge_type_indices, conceptL_idxs, conceptR_idxs):
        if self.One_part:
            return self.forward_ggnn(features, adjacency_matrices, edge_type_indices, conceptL_idxs, conceptR_idxs)
        else:
            return self.forward_gcn_gnn(features, adjacency_matrices, conceptL_idxs, conceptR_idxs)

    def forward_sdgnn(self, features, adjacency_matrices, edge_type_indices, conceptL_idxs, conceptR_idxs):
        hidden_state = features.clone()
        for step in range(self.num_steps):
            edge_type = edge_type_indices[step]
            adjacency_matrix_step = adjacency_matrices[edge_type].toarray() if isinstance(adjacency_matrices[edge_type],
                                                                                          scipy.sparse.spmatrix) else \
            adjacency_matrices[edge_type]
            adjacency_matrix_step = torch.tensor(adjacency_matrix_step, dtype=torch.float32)
            rel_weight = self.compute_relation_weight(edge_type)
            weighted_messages = torch.matmul(adjacency_matrix_step, hidden_state)
            weighted_messages = torch.matmul(weighted_messages, rel_weight)
            update_gate = torch.sigmoid(self.U(weighted_messages))
            reset_gate = torch.sigmoid(self.V(weighted_messages))
            reset_hidden_state = reset_gate * hidden_state
            candidate_hidden_state = self.activation(self.W[edge_type](reset_hidden_state))
            candidate_hidden_state = self.dropout(candidate_hidden_state)
            hidden_state = (1 - update_gate) * hidden_state + update_gate * candidate_hidden_state

        final_node_representation = self.output_projection(hidden_state)
        # 添加 GraphSAGE Mean Aggregator
        neighbor_aggregated = torch.matmul(adjacency_matrix_step, final_node_representation) / \
                              (adjacency_matrix_step.sum(dim=1, keepdim=True) + 1e-10)
        conceptL_embs = neighbor_aggregated[conceptL_idxs]
        conceptR_embs = neighbor_aggregated[conceptR_idxs]
        conceptW_embs = torch.cat((conceptL_embs, conceptR_embs), dim=1)
        if self.use_SiameseNet==True:
            concept_output = self.SiameseNet(conceptL_embs, conceptR_embs)
            return concept_output
        if self.use_SiameseNet==False:
            concept_output = self.No_Siamese_LogSoftMax(self.No_Siamese_Linear(conceptW_embs))
            return concept_output

    def compute_relation_weight(self, edge_type):
        # Compute the weighted sum of base matrices
        weighted_sum = torch.sum(self.bases * self.edge_type_coefficients[edge_type].view(-1, 1, 1), dim=0)
        return weighted_sum
    def forward_gcn_gnn(self, features, adjacency_matrices, conceptL_idxs, conceptR_idxs):
        hidden_state = features.clone()
        adj = torch.stack([torch.tensor(a.toarray(), dtype=torch.float32) for a in adjacency_matrices]).mean(dim=0)

        if self.use_gcn:
            hidden_state = self.gcn_layer(hidden_state, adj)
        if self.use_gnn:
            hidden_state = self.gnn_layer(hidden_state, adj)
        hidden_state = self.GNN_part_Linear(hidden_state)
        conceptL_embs = hidden_state[conceptL_idxs]
        conceptR_embs = hidden_state[conceptR_idxs]
        concept_output = self.SiameseNet(conceptL_embs, conceptR_embs)
        # return self.One_part_Linear_Softmax(self.One_part_Linear(hidden_state))
        return concept_output



class AT_SiameseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, outputdim, num_heads=1, use_isattention=True):
        super(AT_SiameseNet, self).__init__()
        
        self.use_isattention=use_isattention
        self.attention = SelfAttention(hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, outputdim, bias=True),
            nn.LogSoftmax(dim=1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, outputdim, bias=True),
            nn.LogSoftmax(dim=1)
        )


    def forward_once(self, x):
        x = self.fc1(x)
        if self.use_isattention:
            x, _ = self.attention(x)
        return x

    def forward(self, input1, input2):
        # if self.use_isattention==True:
        #     x = self.mhgat(input1, input2)
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        if self.use_isattention:
            supports = [output1, output2]
            supports = torch.cat(supports, dim=1)
            return self.fc3(supports)
        elif self.use_isattention==False:
            supports = [output1, output2, output1 - output2, output1 * output2]
            supports = torch.cat(supports, dim=1)
            return self.fc2(supports)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, x):
        energy = self.projection(x)
        weights = F.softmax(energy, dim=1)
        outputs = weights * x
        return outputs, weights