from d3ems.networks.mlp import mlp
import torch
import torch.nn as nn


class ContinuousQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hid_dim, activation, cuda):

        super(ContinuousQFunction, self).__init__()
        if cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.q1 = mlp([obs_dim + act_dim] + [hid_dim, hid_dim, hid_dim] + [1], activation)
        self.q2 = mlp([obs_dim + act_dim] + [hid_dim, hid_dim, hid_dim] + [1], activation)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.to(self.device)

    def forward(self, obs, act):
        return self.q1(torch.cat([obs, act], dim=-1)), self.q2(torch.cat([obs, act], dim=-1))
