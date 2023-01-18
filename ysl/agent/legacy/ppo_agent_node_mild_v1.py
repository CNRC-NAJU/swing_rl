from operator import concat
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.optim import Adam
from torch_geometric.nn import ChebConv, GATv2Conv, LayerNorm
from torch.nn import BatchNorm1d

import copy
import numpy as np

from utils.solver import *

# from torch_scatter import segment_coo

class Actor(torch.nn.Module):
    def __init__(self, num_features, num_hiddens, num_filters, num_nodes):
        super().__init__()

        self.conv1 = ChebConv(num_features, num_hiddens, num_filters)
        self.batch_norm1 = BatchNorm1d(num_hiddens)
        self.conv2 = ChebConv(num_hiddens, num_hiddens, num_filters)
        # self.batch_norm2 = BatchNorm1d(num_hiddens)
        # self.conv3 = ChebConv(num_hiddens, num_hiddens, num_filters)
        # self.batch_norm3 = BatchNorm1d(num_hiddens)
        # self.conv4 = ChebConv(num_hiddens, num_hiddens, num_filters)
        # self.batch_norm4 = BatchNorm1d(num_hiddens)
        self.final = torch.nn.Linear(num_hiddens, 1)
        self.norm = LayerNorm(num_nodes)
    

    def forward(self, features, adj):

        x, edge_index = features, adj

        x = F.silu(self.batch_norm1(self.conv1(x, edge_index)))
        # x = F.silu(self.conv1(x, edge_index))
        # x = torch.tanh(self.batch_norm2(self.conv2(x, edge_index)))
        x = torch.tanh(self.conv2(x, edge_index))
        # x = F.silu(self.batch_norm3(self.conv3(x, edge_index)))
        # x = torch.tanh(self.batch_norm4(self.conv4(x, edge_index)))
        x = self.norm(self.final(x).view(-1))

        return x


####


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, gain=1)
        # torch.nn.init.uniform_(m.weight, a=-1., b=1.)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=1)
        m.bias.data.fill_(0)

class PPOAgent():
    def __init__(self, num_features=2, num_hiddens=16, num_filters=1, num_nodes=20, device='cuda', lr=5e-4, sigma=0.1):
        self.actor = Actor(num_features, num_hiddens, num_filters, num_nodes).to(device)
        self.actor.apply(init_weights)
        self.actor_old = copy.deepcopy(self.actor).to(device)
        self.optim = Adam(self.actor.parameters(), lr=lr)
        self.actor_losses = []
        self.device = device
        self.sigma = sigma


    def act(self, features, adj, activity):
        amplitude, actions, dist, entropy = self._get_action(features, adj, activity)
        dist_old = self._get_action_old(features, adj, activity)
        
        log_probs = dist.log_prob(actions)
        log_probs_old = dist_old.log_prob(actions).detach()
        ratio = torch.exp(log_probs.sum() - log_probs_old.sum())

        # amplitude[actions==1] *= 1+self.sigma
        amplitude[actions==0] *= 1-self.sigma

        return amplitude, ratio, entropy


    def act_test(self, features, adj, activity):
        amplitude, _, _, _ = self._get_action(features, adj, activity)

        return amplitude


    def update(self, returns, advantages, ratios, ppo_eps, entropies, ent_coeff):
        self.actor.train()
        # self.critic.train()

        l1 = ratios * advantages
        l2 = torch.clamp(ratios, 1 - ppo_eps, 1 + ppo_eps) * advantages
        actor_loss = (- (torch.minimum(l1, l2) + ent_coeff * entropies).sum())
        
        self.optim.zero_grad()
        actor_loss.backward()
        self.optim.step()

        return actor_loss.clone().detach().cpu().numpy()


    def push(self, returns, advantages, ratios, ppo_eps, entropies, ent_coeff):

        l1 = ratios * advantages
        l2 = torch.clamp(ratios, 1 - ppo_eps, 1 + ppo_eps) * advantages
        
        self.actor_losses.append((- (torch.minimum(l1, l2) + ent_coeff * entropies).sum()))


    def update_mean(self):
        self.actor.train()

        actor_loss = sum(self.actor_losses)/len(self.actor_losses)
        
        self.optim.zero_grad()
        actor_loss.backward()
        self.optim.step()

        self.actor_losses = []

        return actor_loss.clone().detach().cpu().numpy()


    def gen_traj_train(self, seed, edge_list, adjm, 
                 init_mask_gen, steady_phase, steady_dphase, init_power, init_activity, init_failed,
                 gamma, h, K, mass, threshold, node, max_step, ep_len):
        rewards = []
        ratios = []
        entropies = []

        mask_gen = np.copy(init_mask_gen)
        phase = np.copy(steady_phase)
        dphase = np.copy(steady_dphase)
        power = np.copy(init_power)
        activity, failed = np.copy(init_activity), np.copy(init_failed)

        # exert perturbation
        activity[seed], failed[seed] = False, True
        feature = torch.tensor([[power[n], phase[n], dphase[n], int(failed[n]), power[n], phase[n], dphase[n], int(failed[n])] for n in range(node)]).float().to(self.device)
        act_gen = activity*mask_gen 
        amplitude, ratio, entropy = self.act(feature, edge_list, act_gen)
        rebal(power, amplitude.clone().detach().cpu().numpy(), act_gen, failed)
        rewards.append(0)
        ratios.append(ratio)
        entropies.append(entropy)

        n_fail = 0
        count = 0
        step = 0

        while 1:
            failed = np.copy(init_failed)
            df = 0
            ks, js = rk4_numba(adjm, power, phase, dphase, gamma, h, K, mass)
            phase += h * js
            dphase += h * ks

            for n in range(len(power)):
                if (dphase[n] > threshold or dphase[n] < -threshold) and activity[n]:
                    failed[n] = True
                    activity[n] = False
                    n_fail += 1
                    df += 1

            act_gen = activity*mask_gen 
            if len(power[act_gen])==0:
                break

            if df:
                feature = torch.column_stack((feature[:,4:], torch.tensor([[power[n], phase[n], dphase[n], int(failed[n])] for n in range(node)]).to(self.device))).float().to(self.device)
                amplitude, ratio, entropy = self.act(feature, edge_list, act_gen)
                rebal(power, amplitude.clone().detach().cpu().numpy(), act_gen, failed)
                rewards[-1] = -df/node
                rewards.append(0)
                ratios.append(ratio)
                entropies.append(entropy)
                count += 1
                ep_len += 1
                step = 0
            else:
                step += 1
            if step > max_step:
                break
        return n_fail, count, rewards, ratios, entropies, ep_len

    def gen_traj_test(self, seed, edge_list, adjm, 
                 init_mask_gen, steady_phase, steady_dphase, init_power, init_activity, init_failed,
                 gamma, h, K, mass, threshold, node, max_step, ep_len):

        mask_gen = np.copy(init_mask_gen)
        phase = np.copy(steady_phase)
        dphase = np.copy(steady_dphase)
        power = np.copy(init_power)
        activity, failed = np.copy(init_activity), np.copy(init_failed)

        # exert perturbation
        activity[seed], failed[seed] = False, True
        feature = torch.tensor([[power[n], phase[n], dphase[n], int(failed[n]), power[n], phase[n], dphase[n], int(failed[n])] for n in range(node)]).float().to(self.device)
        act_gen = activity*mask_gen 
        amplitude = self.act_test(feature, edge_list, act_gen)
        rebal(power, amplitude.clone().detach().cpu().numpy(), act_gen, failed)

        n_fail = 0
        count = 0
        step = 0

        while 1:
            failed = np.copy(init_failed)
            df = 0
            ks, js = rk4_numba(adjm, power, phase, dphase, gamma, h, K, mass)
            phase += h * js
            dphase += h * ks

            for n in range(len(power)):
                if (dphase[n] > threshold or dphase[n] < -threshold) and activity[n]:
                    failed[n] = True
                    activity[n] = False
                    n_fail += 1
                    df += 1

            act_gen = activity*mask_gen 
            if len(power[act_gen])==0:
                break

            if df:
                feature = torch.column_stack((feature[:,4:], torch.tensor([[power[n], phase[n], dphase[n], int(failed[n])] for n in range(node)]).to(self.device))).float().to(self.device)
                amplitude = self.act_test(feature, edge_list, act_gen)
                rebal(power, amplitude.clone().detach().cpu().numpy(), act_gen, failed)
                count += 1
                step = 0
                ep_len += 1
            else:
                step += 1
            if step > max_step:
                break
        return n_fail, count, ep_len


    def _get_action(self, features, adj, activity):
        output = self.actor(features, adj)
        probs = torch.sigmoid(output[activity])
        dist = Bernoulli(probs)
        actions = dist.sample()
    
        # return output.clone().detach(), actions, dist, value, dist.entropy().mean()
        return probs.clone().detach(), actions, dist, dist.entropy().mean()

    def _get_action_old(self, features, adj, activity):
        output = self.actor_old(features, adj)
        probs = torch.sigmoid(output[activity])

        return Bernoulli(probs)
