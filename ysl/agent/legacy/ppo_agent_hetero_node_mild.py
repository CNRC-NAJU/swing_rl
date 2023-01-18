from operator import concat
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.optim import Adam
from torch_geometric.nn import ChebConv, GATv2Conv, LayerNorm, GINConv
from torch.nn import BatchNorm1d

from lib.hetero import *

import copy

# from torch_scatter import segment_coo

class Actor(torch.nn.Module):
    def __init__(self, num_features, num_hiddens, num_filters, num_nodes, num_cores):
        super().__init__()

        self.conv1 = HeteroChebConv(num_features, num_hiddens, num_filters, num_cores, bias=False)
        self.batch_norm1 = BatchNorm1d(num_hiddens)
        self.conv2 = HeteroChebConv(num_hiddens, num_hiddens, num_filters, num_cores, bias=False)
        self.batch_norm2 = BatchNorm1d(num_hiddens)
        # self.conv3 = HeteroChebConv(num_hiddens, num_hiddens, num_filters, num_cores)
        # self.batch_norm3 = BatchNorm1d(num_hiddens)
        # self.conv4 = HeteroChebConv(num_hiddens, num_hiddens, num_filters, num_cores)
        # self.batch_norm4 = BatchNorm1d(num_hiddens)
        self.final = HeteroLinear(num_hiddens, 1, num_cores)
        self.norm = LayerNorm(num_nodes)
    

    def forward(self, features: Tensor, adj: Tensor, segment: Tensor):
        x, edge_index = features, adj

        x = F.silu(self.batch_norm1(self.conv1(x, edge_index, segment)))
        x = torch.tanh(self.batch_norm2(self.conv2(x, edge_index, segment)))
        # x = torch.tanh(self.conv2(x, edge_index, segment))
        # x = F.silu(self.batch_norm3(self.conv3(x, edge_index, segment)))
        # x = torch.tanh(self.batch_norm4(self.conv4(x, edge_index, segment)))
        x = self.norm(self.final(x, segment).view(-1))

        return x


# class Critic(torch.nn.Module):
#     def __init__(self, num_features, num_hiddens, num_filters):
#         super().__init__()

#         self.conv1 = ChebConv(num_features, num_hiddens, num_filters)
#         # self.conv2 = ChebConv(num_hiddens, num_hiddens, num_filters)
#         self.final = torch.nn.Linear(num_hiddens, 1)


#     def forward(self, features, adj):

#         x, edge_index = features, adj

#         x = torch.tanh(self.conv1(x, edge_index))
#         # x = torch.tanh(self.conv2(x, edge_index))
#         x = self.final(x).view(-1)

#         return x


####


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, gain=1)
        # torch.nn.init.uniform_(m.weight, a=-1., b=1.)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=1)
        m.bias.data.fill_(0)

class PPOAgent():
    def __init__(self, num_features=2, num_hiddens=16, num_filters=1, num_cores=1, num_nodes=20, device='cuda', lr=5e-4, sigma=0.1):
        self.actor = Actor(num_features, num_hiddens, num_filters, num_nodes, num_cores).to(device)
        self.actor.apply(init_weights)
        self.actor_old = copy.deepcopy(self.actor).to(device)
        self.optim = Adam(self.actor.parameters(), lr=lr)
        self.actor_losses = []
        self.device = device
        self.sigma = sigma


    def act(self, features, adj, activity, segment):
        amplitude, actions, dist, entropy = self._get_action(features, adj, activity, segment)
        dist_old = self._get_action_old(features, adj, activity, segment)
        
        log_probs = dist.log_prob(actions)
        log_probs_old = dist_old.log_prob(actions).detach()
        ratio = torch.exp(log_probs.sum() - log_probs_old.sum())

        # amplitude[actions==1] *= 1+self.sigma
        amplitude[actions==0] *= 1-self.sigma

        return amplitude, ratio, entropy


    def update(self, returns, advantages, ratios, ppo_eps, entropies, ent_coeff):
        self.actor.train()

        l1 = ratios * advantages
        l2 = torch.clamp(ratios, 1 - ppo_eps, 1 + ppo_eps) * advantages
        actor_loss = (- (torch.minimum(l1, l2) + ent_coeff * entropies).sum())
        # actor_loss = (- (torch.minimum(l1, l2) + ent_coeff * entropies).mean())
        
        self.optim.zero_grad()
        actor_loss.backward()
        self.optim.step()

        return actor_loss.clone().detach().cpu().numpy()


    def push(self, returns, advantages, ratios, ppo_eps, entropies, ent_coeff):

        l1 = ratios * advantages
        l2 = torch.clamp(ratios, 1 - ppo_eps, 1 + ppo_eps) * advantages
        
        self.actor_losses.append((- (torch.minimum(l1, l2) + ent_coeff * entropies).sum()))
        # self.actor_losses.append((- (torch.minimum(l1, l2) + ent_coeff * entropies).mean()))


    def update_mean(self):
        self.actor.train()

        actor_loss = sum(self.actor_losses)/len(self.actor_losses)
        
        self.optim.zero_grad()
        actor_loss.backward()
        self.optim.step()

        self.actor_losses = []

        return actor_loss.clone().detach().cpu().numpy()


    def act_test(self, features, adj): # for test
        amplitude = self._get_action_test(features, adj)

        return amplitude


    def _get_action(self, features, adj, activity, segment):
        output = self.actor(features, adj, segment)
        probs = torch.sigmoid(output[activity])
        dist = Bernoulli(probs)
        actions = dist.sample()
    
        return probs.clone().detach(), actions, dist, dist.entropy().mean()

    def _get_action_old(self, features, adj, activity, segment):
        output = self.actor_old(features, adj, segment)
        probs = torch.sigmoid(output[activity])

        return Bernoulli(probs)

    def _get_action_test(self, features, adj):
        output = self.actor(features, adj)
    
        return torch.sigmoid(output)