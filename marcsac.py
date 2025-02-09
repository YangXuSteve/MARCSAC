from copy import deepcopy
import numpy as np

from d3ems.networks.actor import SquashedGaussianActor
from d3ems.networks.critic import ContinuousQFunction
from d3ems.networks.lam import Lam
import torch
import torch.nn as nn


def pack_actions_tensor(*actions):
    ga = torch.cat(list(actions), dim=-1)
    return ga


class MARCSAC(nn.Module):
    def __init__(self, args):
        super(MARCSAC, self).__init__()
        self.args = args

        self.agent_num = args.agent_num
        self.action_highs = args.action_highs
        self.action_lows = args.action_lows
        self.obs_dims = args.observation_dims
        self.action_dims = args.action_dims
        self.hid_size = args.hid_size

        self.gamma = args.gamma
        self.alpha = args.alpha

        if args.hid_activation == "relu":
            self.hid_activation = nn.ReLU
        elif args.hid_activation == "tanh":
            self.hid_activation = nn.Tanh
        else:
            raise Exception("No specified activation")

        if args.optim == "adam":
            self.optimizer_class = torch.optim.Adam
        elif args.optim == "rmsprop":
            self.optimizer_class = torch.optim.RMSprop
        else:
            raise Exception("No specified optimizer")

        self.cuda = args.cuda
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.pis = [SquashedGaussianActor(self.obs_dims[i], self.action_dims[i], self.hid_size, self.hid_activation, self.action_highs[i], self.action_lows[i], self.cuda)
                    for i in range(self.agent_num)]
        self.central_q = ContinuousQFunction(sum(self.obs_dims), sum(self.action_dims), self.hid_size, self.hid_activation, self.cuda)
        self.cqs = [ContinuousQFunction(sum(self.obs_dims), sum(self.action_dims), self.hid_size, self.hid_activation, self.cuda)
                    for i in range(self.agent_num)]
        self.lams = [Lam(sum(self.obs_dims), self.hid_size, self.hid_activation, self.args.max_lam, self.cuda) for i in range(self.agent_num)]

        if args.init_type == "normal":
            for i in range(self.agent_num):
                for layer in self.pis[i].net:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, 0.0, self.args.init_std)
                nn.init.normal_(self.pis[i].mu_layer.weight, 0.0, self.args.init_std)
                nn.init.normal_(self.pis[i].log_std_layer.weight, 0.0, self.args.init_std)
                for layer in self.cqs[i].q1:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, 0.0, self.args.init_std)
                for layer in self.cqs[i].q2:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, 0.0, self.args.init_std)
                for layer in self.lams[i].net:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, 0.0, self.args.init_std)
            for layer in self.central_q.q1:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0.0, self.args.init_std)
            for layer in self.central_q.q2:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0.0, self.args.init_std)
        elif args.init_type == "orthogonal":
            for i in range(self.agent_num):
                for layer in self.pis[i].net:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(self.args.hid_activation))
                nn.init.orthogonal_(self.pis[i].mu_layer.weight, gain=nn.init.calculate_gain(self.args.hid_activation))
                nn.init.orthogonal_(self.pis[i].log_std_layer.weight, gain=nn.init.calculate_gain(self.args.hid_activation))
                for layer in self.cqs[i].q1:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(self.args.hid_activation))
                for layer in self.cqs[i].q2:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(self.args.hid_activation))
                for layer in self.lams[i].net:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(self.args.hid_activation))
            for layer in self.central_q.q1:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(self.args.hid_activation))
            for layer in self.central_q.q2:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(self.args.hid_activation))
        else:
            raise Exception("No specified init type")

        self.central_q_targ = deepcopy(self.central_q)
        self.cq_targs = deepcopy(self.cqs)
        for p in self.central_q_targ.parameters():
            p.requires_grad = False
        for cq_targ in self.cq_targs:
            for p in cq_targ.parameters():
                p.requires_grad = False

        self.q_parameters = list()
        self.cq_parameters = list()
        self.lam_parameters = list()
        for i in range(self.agent_num):
            self.cq_parameters += list(self.cqs[i].parameters())
            self.lam_parameters += list(self.lams[i].parameters())
        self.q_parameters += list(self.central_q.parameters())

        self.pi_optimizers = [self.optimizer_class(self.pis[i].parameters(), lr=args.policy_lrate) for i in range(self.agent_num)]
        self.q_optimizer = self.optimizer_class(self.q_parameters, lr=args.value_lrate)
        self.cq_optimizer = self.optimizer_class(self.cq_parameters, lr=args.value_lrate)
        self.lam_optimizer = self.optimizer_class(self.lam_parameters, lr=args.lam_lrate)

        self.policy_end_lrate = args.policy_end_lrate
        self.value_end_lrate = args.value_end_lrate
        self.lam_end_lrate = args.lam_end_lrate

        self.pi_schedulers = [torch.optim.lr_scheduler.ExponentialLR(self.pi_optimizers[i], 0.999) for i in range(self.agent_num)]
        self.q_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.q_optimizer, 0.999)
        self.cq_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.cq_optimizer, 0.999)
        self.lam_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.lam_optimizer, 0.999)

        self.target_lr = args.target_lr
        self.to(self.device)

    def _compute_loss_q(self, data):
        obss, acts, obs2s, done = data['obss'], data['acts'], data['obs2s'], data['done']
        global_obs = pack_actions_tensor(*obss).to(self.device)
        global_obs2 = pack_actions_tensor(*obs2s).to(self.device)
        global_act = pack_actions_tensor(*acts).to(self.device)
        rew = data['rew']

        with torch.no_grad():
            act2s_info = [self.pis[j](obs2s[j]) for j in range(self.agent_num)]
            act2s = [act2s_info[j][0] for j in range(self.agent_num)]
            logp_act2s = [act2s_info[j][1] for j in range(self.agent_num)]
            global_act2 = pack_actions_tensor(*act2s).to(self.device)
            entropy = -self.alpha * torch.cat(logp_act2s, dim=-1).to(self.device).sum(dim=-1, keepdim=True)

            q1_target, q2_target = self.central_q_targ(global_obs2, global_act2)
            q_target = torch.min(q1_target, q2_target)
            target_q = rew + self.gamma * (1 - done) * (q_target + entropy)

        q1_eval, q2_eval = self.central_q(global_obs, global_act)

        loss_q = nn.MSELoss()(q1_eval, target_q) + nn.MSELoss()(q2_eval, target_q)

        return loss_q

    def _compute_loss_cq(self, data):
        loss_cq = 0
        cq1_target, cq2_target = list(range(self.agent_num)), list(range(self.agent_num))
        cq_target, target_cq = list(range(self.agent_num)), list(range(self.agent_num))
        cq1_eval, cq2_eval = list(range(self.agent_num)), list(range(self.agent_num))

        obss, acts, obs2s, done = data['obss'], data['acts'], data['obs2s'], data['done']
        global_obs = pack_actions_tensor(*obss).to(self.device)
        global_obs2 = pack_actions_tensor(*obs2s).to(self.device)
        global_act = pack_actions_tensor(*acts).to(self.device)
        costs = data['costs']

        with torch.no_grad():
            act2s_info = [self.pis[j](obs2s[j]) for j in range(self.agent_num)]
            act2s = [act2s_info[j][0] for j in range(self.agent_num)]
            global_act2 = pack_actions_tensor(*act2s).to(self.device)
            for i in range(self.agent_num):
                cq1_target[i], cq2_target[i] = self.cq_targs[i](global_obs2, global_act2)
                cq_target[i] = torch.min(cq1_target[i], cq2_target[i])
                target_cq[i] = (1 - self.gamma) * costs[i] + self.gamma * torch.max(costs[i], (1 - done) * cq_target[i])

        for i in range(self.agent_num):
            cq1_eval[i], cq2_eval[i] = self.cqs[i](global_obs, global_act)
            loss_cq += nn.MSELoss()(cq1_eval[i], target_cq[i]) + nn.MSELoss()(cq2_eval[i], target_cq[i])

        return loss_cq

    def _compute_loss_pi(self, i, data):
        obss = data['obss']
        global_obs = pack_actions_tensor(*obss).to(self.device)

        acts_info = [self.pis[j](obss[j]) for j in range(self.agent_num)]
        acts = [acts_info[j][0] for j in range(self.agent_num)]
        logp_acts = [acts_info[j][1] for j in range(self.agent_num)]
        entropy = -self.alpha * logp_acts[i]
        global_act = pack_actions_tensor(*acts).to(self.device)

        q1_eval, q2_eval = self.central_q(global_obs, global_act)
        q_eval = torch.min(q1_eval, q2_eval) + entropy

        cq1_eval, cq2_eval = self.cqs[i](global_obs, global_act)
        cq_eval = torch.min(cq1_eval, cq2_eval)

        loss_pi = (-q_eval + self.lams[i](global_obs) * cq_eval).mean()

        return loss_pi

    def _compute_loss_lam(self, data):
        loss_lam = 0
        cq1_eval, cq2_eval = list(range(self.agent_num)), list(range(self.agent_num))
        cq_eval = list(range(self.agent_num))

        obss = data['obss']
        global_obs = pack_actions_tensor(*obss).to(self.device)

        acts_info = [self.pis[j](obss[j]) for j in range(self.agent_num)]
        acts = [acts_info[j][0] for j in range(self.agent_num)]
        global_act = pack_actions_tensor(*acts).to(self.device)

        for i in range(self.agent_num):
            cq1_eval[i], cq2_eval[i] = self.cqs[i](global_obs, global_act)
            cq_eval[i] = torch.min(cq1_eval[i], cq2_eval[i])
            loss_lam += (-self.lams[i](global_obs) * cq_eval[i]).mean()
            # print(f'Agent {i}, Lam {self.lams[i](global_obs)}, Cq: {cq_eval[i]}')

        return loss_lam

    def update_q(self, batch):
        for i in range(self.agent_num):
            for p in self.pis[i].parameters():
                p.requires_grad = False
        for p in self.lam_parameters:
            p.requires_grad = False

        self.q_optimizer.zero_grad()
        loss_q = self._compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()

        self.cq_optimizer.zero_grad()
        loss_cq = self._compute_loss_cq(batch)
        loss_cq.backward()
        self.cq_optimizer.step()

        for i in range(self.agent_num):
            for p in self.pis[i].parameters():
                p.requires_grad = True
        for p in self.lam_parameters:
            p.requires_grad = True

        self._polyak_target_update()

        if self.q_optimizer.param_groups[0]['lr'] >= self.value_end_lrate:
            self.q_scheduler.step()

        if self.cq_optimizer.param_groups[0]['lr'] >= self.value_end_lrate:
            self.cq_scheduler.step()

        for i in range(self.agent_num):
            if self.pi_optimizers[i].param_groups[0]['lr'] >= self.policy_end_lrate:
                self.pi_schedulers[i].step()

        if self.lam_optimizer.param_groups[0]['lr'] >= self.lam_end_lrate:
            self.lam_scheduler.step()

        return loss_q, loss_cq

    def update_pi(self, batch):
        loss_pi = list(range(self.agent_num))

        for p in self.q_parameters:
            p.requires_grad = False
        for p in self.cq_parameters:
            p.requires_grad = False
        for p in self.lam_parameters:
            p.requires_grad = False
        for i in range(self.agent_num):
            for p in self.pis[i].parameters():
                p.requires_grad = False

        for i in range(self.agent_num):
            for p in self.pis[i].parameters():
                p.requires_grad = True
            self.pi_optimizers[i].zero_grad()
            loss_pi[i] = self._compute_loss_pi(i, batch)
            loss_pi[i].backward()
            self.pi_optimizers[i].step()
            for p in self.pis[i].parameters():
                p.requires_grad = False

        for p in self.q_parameters:
            p.requires_grad = True
        for p in self.cq_parameters:
            p.requires_grad = True
        for p in self.lam_parameters:
            p.requires_grad = True
        for i in range(self.agent_num):
            for p in self.pis[i].parameters():
                p.requires_grad = True

        return loss_pi

    def update_lam(self, batch):
        for p in self.q_parameters:
            p.requires_grad = False
        for p in self.cq_parameters:
            p.requires_grad = False
        for i in range(self.agent_num):
            for p in self.pis[i].parameters():
                p.requires_grad = False

        self.lam_optimizer.zero_grad()
        loss_lam = self._compute_loss_lam(batch)
        loss_lam.backward()
        self.lam_optimizer.step()

        for p in self.q_parameters:
            p.requires_grad = True
        for p in self.cq_parameters:
            p.requires_grad = True
        for i in range(self.agent_num):
            for p in self.pis[i].parameters():
                p.requires_grad = True

        return loss_lam

    def predict(self, i, observation, deterministic=False, with_logprob=False):
        if hasattr(observation, 'numpy'):
            observation = observation.numpy()
        else:
            observation = np.array(observation)

        observation = observation.reshape((-1,) + (self.obs_dims[i],))
        observation = torch.as_tensor(observation, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action, log_prob = self._predict(i, observation, deterministic=deterministic, with_logprob=with_logprob)

        action = action.cpu().numpy().squeeze()

        if log_prob is not None:
            log_prob = log_prob.cpu().numpy().squeeze()

        action = np.clip(action, self.action_lows[i], self.action_highs[i])

        if with_logprob:
            return action, log_prob
        else:
            return action

    def _predict(self, i, observation, deterministic=False, with_logprob=False):
        action, log_prob = self.pis[i](observation, deterministic, with_logprob)

        return action, log_prob

    def forward(self, i, obs):
        self._predict(i, obs, True)

    def _polyak_target_update(self):
        for param, target_param in zip(self.central_q.parameters(), self.central_q_targ.parameters()):
            target_param.data.copy_(self.target_lr * param.data + (1 - self.target_lr) * target_param.data)
        for i in range(self.agent_num):
            for param, target_param in zip(self.cqs[i].parameters(), self.cq_targs[i].parameters()):
                target_param.data.copy_(self.target_lr * param.data + (1 - self.target_lr) * target_param.data)

    def save_to_pth(self, path):
        state = {'pis': [self.pis[i].state_dict() for i in range(self.agent_num)],
                 'central_q': self.central_q.state_dict(),
                 'central_q_targ': self.central_q_targ.state_dict(),
                 'cqs': [self.cqs[i].state_dict() for i in range(self.agent_num)],
                 'cq_targets': [self.cq_targs[i].state_dict() for i in range(self.agent_num)],
                 'lams': [self.lams[i].state_dict() for i in range(self.agent_num)]}
        torch.save(state, path)

    def load_from_pth(self, path):
        saved_model = torch.load(path)
        for i in range(self.agent_num):
            self.pis[i].load_state_dict(saved_model['pis'][i])
            self.cqs[i].load_state_dict(saved_model['cqs'][i])
            self.cq_targs[i].load_state_dict(saved_model['cq_targets'][i])
            self.lams[i].load_state_dict(saved_model['lams'][i])
        self.central_q.load_state_dict(saved_model['central_q'])
        self.central_q_targ.load_state_dict(saved_model['central_q_targ'])
