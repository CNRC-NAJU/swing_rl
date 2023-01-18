import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch_geometric.nn import ChebConv

import copy

# from torch_scatter import segment_coo

class Actor(torch.nn.Module):
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
        # x = torch.sigmoid(self.final(x).view(-1))

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
        # torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
        # m.bias.data.fill_(0)

class PPOAgent():
    def __init__(self, num_features=2, num_hiddens=16, num_filters=1, device='cuda'):
        self.actor = Actor(num_features, num_hiddens, num_filters).to(device)
        self.critic = Critic(num_features, num_hiddens, num_filters).to(device)
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.actor_old = copy.deepcopy(self.actor).to(device)
        self.optim = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=1e-4)


    def act_for_n(self, features, adjs, nn_idxes, n_idxes):
        actions, dists, values, entropy = self._get_action_n(features, adjs, nn_idxes, n_idxes)
        dists_old = self._get_action_n_old(features, adjs, nn_idxes)
        
        log_probs = torch.stack([sum([dists[n].log_prob(actions[n][m]) for m in range(len(actions[n]))]) for n in range(len(features)) if features[n]!=None])
        log_probs_old = torch.stack([sum([dists_old[n].log_prob(actions[n][m]).detach() for m in range(len(actions[n]))]) for n in range(len(features)) if features[n]!=None])
        ratio = torch.exp(log_probs - log_probs_old)

        return [[a.item() for a in action] if action!=None else None for action in actions], ratio, values, entropy


    def update_mean(self, returns, advantages, ratios, values, ppo_eps, entropies, ent_coeff, num_agents):
        self.actor.train()
        self.critic.train()
        actor_loss = []
        critic_loss = []

        for n in range(num_agents):
            returns[n] = returns[n].detach()
            advantages[n] = advantages[n].detach()
            entropies[n] = entropies[n].detach()

            l1 = ratios[n] * advantages[n]
            l2 = torch.clamp(ratios[n], 1 - ppo_eps, 1 + ppo_eps) * advantages[n]

            # actor_loss = - (torch.minimum(l1, l2).sum() + ent_coeff * entropies.mean())
            # actor_loss = - torch.minimum(l1, l2).sum() 
            actor_loss.append(- (torch.minimum(l1, l2) + ent_coeff * entropies[n]).sum())
            critic_loss.append(F.smooth_l1_loss(returns[n], values[n]).mean())
        
        self.optim.zero_grad()
        (sum(actor_loss)/len(actor_loss)).backward()
        (sum(critic_loss)/len(critic_loss)).backward()
        self.optim.step()

        return sum(actor_loss).clone().detach().cpu().numpy()/len(actor_loss), sum(critic_loss).clone().detach().cpu().numpy()/len(critic_loss)


    def _get_action_n(self, features, adjs, idxes, n_idxes):
        output, value = [], []
        for n in range(len(features)):
            if features[n]!=None:
                output.append(self.actor(features[n], adjs[n]))
                value.append(self.critic(features[n], adjs[n]))
            else:
                output.append(None)
                value.append(None)
                
        outputs = [F.softmax(output[n][idxes[n]], dim=0) if features[n]!=None else None for n in range(len(features))]
        values = [value[n][n_idxes[n]] if features[n]!=None else None for n in range(len(features))]
        dists = [Categorical(output) if output!=None else None for output in outputs]
        actions = [[dist.sample() for _ in range(10)] if dist!=None else None for dist in dists]
        entropy = [dist.entropy() if dist!=None else None for dist in dists]
        
        return actions, dists, values, entropy

    def _get_action_n_old(self, features, adjs, idxes):
        output = []
        for n in range(len(features)):
            if features[n]!=None:
                output.append(self.actor_old(features[n], adjs[n]))
            else:
                output.append(None)
        outputs = [F.softmax(output[n][idxes[n]], dim=0) if features[n]!=None else None for n in range(len(features))]
        dists = [Categorical(output) if output!=None else None for output in outputs]
        
        return dists