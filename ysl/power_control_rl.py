import numpy as np
import networkx as nx

from torch_geometric.utils import from_networkx
import torch

from agent.ppo_agent import *

from utils.utils import *
from utils.logger import *
from utils.solver import *
from utils.load_data import *

import copy


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 123

# init config

protocol = 5
lossless = True
g, adjm, alpha, mask_gen, phase, dphase, power, mass, gamma, _, _, mass_min = load_uk(
    protocol=protocol, lossless=lossless
)
net_type = f"uk w real params, protocol_{protocol}"
data = from_networkx(g.to_undirected())
edge_list, edge_weight = data.edge_index.to(device), data.weight.to(device) / 100
# edge_list, edge_weight = data.edge_index.to(device), torch.ones((g.number_of_edges()*2, )).to(device)
node = g.number_of_nodes()

# g, adjm, mask_gen, phase, dphase, power, mass, gamma, _, net_type = load_shk(case=5, seed=seed) # case=2 for r=1, case=3 for r=10
# # net_type = 'shk w K=4 case 3'
# data = from_networkx(g)
# edge_list, edge_weight = data.edge_index.to(device), torch.ones((g.number_of_edges()*2, )).to(device)
# node = g.number_of_nodes()
# alpha = np.zeros((node, node))
# lossless = True
# m0 = 0.01


threshold = 0.02
threshold *= np.pi * 2

h = 0.001
m0 = mass_min  # how much mass we want to leave

while 1:
    ks, js = rk4_numba(adjm, power, phase, dphase, gamma, h, 1, mass)
    phase += h * js
    dphase += h * ks
    if check_vel(dphase):
        break
steady_phase = np.copy(phase)
steady_dphase = np.copy(dphase)
init_power = np.copy(power)
init_mass = np.copy(mass)


# train
ppo_eps = 0.1
entropy_coeff = 1e-4
lr = 5e-4
num_features = 12
num_hiddens = 32
num_filters = 2
train_if_fails = True
cum_rew = False
reward_weight = False

# exploration = 0.25
exploration = 0.5
# exploration = 0.75
# exploration = 0.99
update_every = 4
discount_factor = 0.99

torch.random.manual_seed(seed)
agent = PPOAgent(
    num_features=num_features,
    num_hiddens=num_hiddens,
    num_filters=num_filters,
    num_nodes=node,
    device=device,
    lr=lr,
    sigma=exploration,
)

num_episodes = 300000
max_step = 1 / h

params = {
    "ConvType": "Cheb silu tanh norm sigmoid version 2",
    "NetworkType": net_type,
    "NumNodes": node,
    "NumFeatures": num_features,
    "NumHiddens": num_hiddens,
    "NumFilters": num_filters,
    "LR": lr,
    "EntropyCoeff": entropy_coeff,
    "TrainIfFails": train_if_fails,
    "CumRew": cum_rew,
    "RewWei": reward_weight,
    "PPOEps": ppo_eps,
    "Exploration": exploration,
    "Threshold": threshold / (np.pi * 2),
    "deltaT": h,
    "Lossless": lossless,
    "UpdateEvery": update_every,
    "InactiveMass": m0,
    "DiscountFactor": discount_factor,
    "Seed": seed,
}

logger = MetricLogger(params)
init_mask_gen = np.copy(mask_gen)
init_activity, init_failed = np.full((node,), True, dtype=bool), np.full(
    (node,), False, dtype=bool
)


# train and test
min_n_fail_mean = node
for ep in range(num_episodes):
    if ep % 10 == 0:
        ep_len = 0
        with torch.no_grad():
            agent.actor.eval()
            n_fail_mean = 0
            for seed in range(node):
                if not mask_gen[seed]:
                    continue
                n_fail, count, ep_len = agent.gen_traj_test(
                    seed,
                    edge_list,
                    edge_weight,
                    adjm,
                    alpha,
                    init_mask_gen,
                    steady_phase,
                    steady_dphase,
                    init_power,
                    init_activity,
                    init_failed,
                    gamma,
                    h,
                    1,
                    init_mass,
                    threshold,
                    node,
                    max_step,
                    ep_len,
                    m0,
                )
                n_fail_mean += n_fail
                logger.log_step(n_fail / node, 0, count)
            n_fail_mean /= node
            if n_fail_mean < min_n_fail_mean:
                torch.save(agent.actor.state_dict(), logger.save_dir / "actor.pt")
                min_n_fail_mean = n_fail_mean
        logger.log_episode(0, 0)
        logger.record(ep + 1, step=ep_len)
    else:
        agent.actor.train()
        ep_len = 0
        seeds = [n for n in range(node)]
        np.random.shuffle(seeds)
        for seed in seeds:
            if not mask_gen[seed]:
                continue
            n_fail, count, rewards, ratios, entropies, ep_len = agent.gen_traj_train(
                seed,
                edge_list,
                edge_weight,
                adjm,
                alpha,
                init_mask_gen,
                steady_phase,
                steady_dphase,
                init_power,
                init_activity,
                init_failed,
                gamma,
                h,
                1,
                init_mass,
                threshold,
                node,
                max_step,
                ep_len,
                m0,
            )
            if n_fail or (not train_if_fails):
                ratios = torch.stack(ratios)
                entropies = torch.stack(entropies)

                returns = calculate_returns(rewards, discount_factor).to(device)
                agent.push(returns, returns, ratios, ppo_eps, entropies, entropy_coeff)
                logger.log_step(n_fail / node, 0, count)
                # if reward_weight:
                #     agent.actor_losses[-1] *= (n_fail+1)/node
            if len(agent.actor_losses) == update_every:
                agent.actor_old = copy.deepcopy(agent.actor)
                actor_loss = agent.update_mean()
        logger.log_episode(0, 0)
        logger.record(ep + 1, step=ep_len)
