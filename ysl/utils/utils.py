import torch

def calculate_returns(rewards, discount_factor):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    return torch.tensor(returns).float()

def calculate_advantages(returns, values):
    advantages = returns - values

    return advantages
