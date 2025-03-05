import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class SDM(nn.Module):
    def __init__(self, num_node, input_dim, latent_dim1, output_dim, drop_prob=0.0, acti_fn=nn.ReLU(), bias=True):
        super().__init__()
        self.num_node = num_node
        self.fc11 = Dense(input_dim, latent_dim1, drop_prob, acti_fn, bias)
        self.fc12 = Dense(latent_dim1, output_dim, drop_prob, acti_fn, bias)
        self.TopM_adj = Top_M(self.num_node, 2)


    def forward(self, adj):
        """
        call encoder here
        """
        Z_A = self.fc12(self.fc11(adj))
        A_e = self.topk_activ_adj(Z_A)
        return A_e

class Top_M(torch.nn.Module):
    def __init__(self,k,t):
        super(Top_M, self).__init__()
        self.k = k
        self.t = t

    def forward(self, x):
        m = torch.distributions.gumbel.Gumbel(0,1)
        z = m.sample(torch.tensor(x.size()))
        w = torch.log(x+1e-30)
        keys = w + z
        onehot_approx = torch.zeros_like(x)
        khot_list = None
        for i in range(self.k):
            khot_mask = torch.maximum(1 - onehot_approx, torch.full(x.size(),1e-20))
            keys += torch.log(khot_mask)
            onehot_approx = F.softmax(keys / self.t, dim=-1)
            if khot_list is None:
                khot_list = onehot_approx.clone()
            else:
                khot_list += onehot_approx
        return khot_list