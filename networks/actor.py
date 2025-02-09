from d3ems.networks.mlp import mlp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hid_dim, activation, act_high, act_low, cuda):

        super(SquashedGaussianActor, self).__init__()
        if cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.net = mlp([obs_dim] + [hid_dim, hid_dim, hid_dim], activation, activation)
        self.mu_layer = nn.Linear(hid_dim, act_dim)
        self.log_std_layer = nn.Linear(hid_dim, act_dim)
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=self.device)
        self.act_low = torch.tensor(act_low, dtype=torch.float32, device=self.device)
        self.to(self.device)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1, keepdim=True)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = 0.5 * (self.act_high - self.act_low) * pi_action + 0.5 * (self.act_high + self.act_low)

        return pi_action, logp_pi
