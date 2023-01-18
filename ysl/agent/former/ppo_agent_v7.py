from operator import concat
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.optim import Adam
from torch_geometric.nn import ChebConv, GATv2Conv, LayerNorm

import copy

# from torch_scatter import segment_coo

class Actor(torch.nn.Module):
    def __init__(self, num_features, num_hiddens, num_filters):
        super().__init__()

        self.conv1 = ChebConv(num_features, num_hiddens, num_filters)
        # self.norm1 = LayerNorm(num_hiddens)
        # self.conv2 = ChebConv(num_hiddens, num_hiddens, num_filters)
        # self.norm2 = LayerNorm(num_hiddens)
        self.final = torch.nn.Linear(num_hiddens, 1)

        self.num_hiddens = num_hiddens
    

    def forward(self, features, adj):

        x, edge_index = features, adj

        # x = torch.tanh(self.norm1(self.conv1(x, edge_index)))
        x = torch.tanh(self.conv1(x, edge_index))
        # x = torch.tanh(self.conv2(x, edge_index))
        # x = self.norm2(x)
        x = self.final(x).view(-1)
        # x = F.softsign(self.final(x).view(-1))
        # x = torch.sigmoid(self.final(x))

        return x


class Critic(torch.nn.Module):
    def __init__(self, num_features, num_hiddens, num_filters):
        super().__init__()

        self.conv1 = ChebConv(num_features, num_hiddens, num_filters)
        # self.conv2 = ChebConv(num_hiddens, num_hiddens, num_filters)
        self.final = torch.nn.Linear(num_hiddens, 1)

        self.num_hiddens = num_hiddens


    def forward(self, features, adj):

        x, edge_index = features, adj

        x = torch.tanh(self.conv1(x, edge_index))
        # x = torch.tanh(self.conv2(x, edge_index))
        x = self.final(x).view(-1)

        return x


####


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, gain=1.)
        # uniform_(m.weight, a=-1., b=1.)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        # m.bias.data.fill_(0)

class PPOAgent():
    def __init__(self, num_features=2, num_hiddens=16, num_filters=1, device='cuda', lr=5e-4):
        self.actor = Actor(num_features, num_hiddens, num_filters).to(device)
        self.critic = Critic(num_features, num_hiddens, num_filters).to(device)
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.actor_old = copy.deepcopy(self.actor).to(device)
        self.optim = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.actor_losses = []
        self.critic_losses = []


    def act(self, features, adj, activity):
        amplitude, actions, dist, value, entropy = self._get_action(features, adj, activity)
        dist_old = self._get_action_old(features, adj, activity)
        
        log_probs = dist.log_prob(actions)
        log_probs_old = dist_old.log_prob(actions).detach()
        ratio = torch.exp(log_probs.sum() - log_probs_old.sum())

        return amplitude, actions>0, ratio, value, entropy


    def update(self, returns, advantages, ratios, values, ppo_eps, entropies, ent_coeff):
        self.actor.train()
        self.critic.train()

        l1 = ratios * advantages
        l2 = torch.clamp(ratios, 1 - ppo_eps, 1 + ppo_eps) * advantages
        actor_loss = (- (torch.minimum(l1, l2) + ent_coeff * entropies).sum())
        critic_loss = F.smooth_l1_loss(returns, values).mean()
        
        self.optim.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optim.step()

        return actor_loss.clone().detach().cpu().numpy(), critic_loss.clone().detach().cpu().numpy()


    def push(self, returns, advantages, ratios, values, ppo_eps, entropies, ent_coeff):

        l1 = ratios * advantages
        l2 = torch.clamp(ratios, 1 - ppo_eps, 1 + ppo_eps) * advantages
        
        self.actor_losses.append((- (torch.minimum(l1, l2) + ent_coeff * entropies).sum()))
        self.critic_losses.append(F.smooth_l1_loss(returns, values).mean())


    def update_mean(self):
        self.actor.train()
        self.critic.train()

        actor_loss = sum(self.actor_losses)/len(self.actor_losses)
        critic_loss = sum(self.critic_losses)/len(self.critic_losses)
        
        self.optim.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optim.step()

        self.actor_losses = []
        self.critic_losses = []

        return actor_loss.clone().detach().cpu().numpy(), critic_loss.clone().detach().cpu().numpy()


    def act_test(self, features, adj): # for test
        amplitude = self._get_action_test(features, adj)

        return amplitude


    def _get_action(self, features, adj, activity):
        output, value = self.actor(features, adj), self.critic(features, adj).mean()
        probs = F.sigmoid(output)
        dist = Bernoulli(probs[activity])
        actions = dist.sample()
    
        # return probs.clone().detach(), actions, dist, value, dist.entropy().mean()
        return F.softplus(output.clone().detach()), actions, dist, value, dist.entropy().mean()

    def _get_action_old(self, features, adj, activity):
        output = self.actor_old(features, adj)
        probs = F.sigmoid(output)
        dist = Bernoulli(probs[activity])

        return dist

    def _get_action_test(self, features, adj):
        output = self.actor(features, adj)
    
        return output