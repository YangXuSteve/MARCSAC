from d3ems.networks.mlp import mlp
import torch
import torch.nn as nn
import torch.nn.functional as F


class Lam(nn.Module):
	def __init__(self, obs_dim, hid_dim, activation, max_lam, cuda):
		super(Lam, self).__init__()
		if cuda:
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")
		self.max_lam = max_lam
		self.net = mlp([obs_dim] + [hid_dim, hid_dim, hid_dim] + [1], activation)
		self.to(self.device)

	def forward(self, obs):
		out = self.net(obs)
		lam = F.softplus(out)
		lam = torch.clamp(lam, 0.0, self.max_lam)

		return lam
