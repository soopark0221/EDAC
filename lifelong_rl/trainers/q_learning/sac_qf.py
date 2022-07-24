import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from collections import OrderedDict

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.torch.distributions import TanhNormal
from lifelong_rl.util.eval_util import create_stats_ordered_dict
from lifelong_rl.torch.pytorch_util import np_to_pytorch_batch


def train_qfs(batch, policy_list, qfs, target_qfs, alpha):
    batch = np_to_pytorch_batch(batch)

    discount=0.99  # Discount factor
    reward_scale=1.0  # Scaling of rewards to modulate entropy bonus
    qf_lr=0.0003  # Learning rate of Q functions
    optimizer_class=optim.Adam  # Class of optimizer for all networks
    soft_target_tau=5e-3  # Rate of update of target networks
    target_update_period=1 # How often to update target networks
    max_q_backup=False
    deterministic_backup=False
    num_qs=10
    qf_criterion = nn.MSELoss(reduction='none')
    eta = 1.0

    qfs_optimizer = optimizer_class(
            qfs.parameters(),
            lr=qf_lr                    
        )

    obs= batch['observations']
    next_obs = batch['next_observations']
    actions = batch['actions']
    rewards = batch['rewards']
    terminals = batch['terminals']
               
    new_next_actions_list, new_log_pi_list, new_pi_list = [], [], []
        
    for policy in policy_list:
        new_next_actions, _, _, new_log_pi, *_ = policy(
                next_obs,
                reparameterize=False,
                return_log_prob=True,
                )
        new_next_actions_list.append(new_next_actions)
        new_log_pi_list.append(new_log_pi)
        new_pi_list.append(torch.exp(new_log_pi))

    qs_pred = qfs(obs, actions)
    new_next_actions = torch_average(new_next_actions_list)
    new_log_pi = torch_average(new_log_pi_list)

    #new_pi = torch_average(new_pi_list)
    #new_log_pi = torch.log(new_pi)

    target_q_values = target_qfs.sample(next_obs, new_next_actions)
    target_q_values -= alpha * new_log_pi

    future_values = (1. - terminals) * discount * target_q_values
    q_target = reward_scale * rewards + future_values
    qfs_loss = qf_criterion(qs_pred, q_target.detach().unsqueeze(0))
    qfs_loss = qfs_loss.mean(dim=(1, 2)).sum()

    qfs_loss_total = qfs_loss

    if eta > 0:
        obs_tile = obs.unsqueeze(0).repeat(num_qs, 1, 1)
        actions_tile = actions.unsqueeze(0).repeat(num_qs, 1, 1).requires_grad_(True)
        qs_preds_tile = qfs(obs_tile, actions_tile)
        qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
        qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
        qs_pred_grads = qs_pred_grads.transpose(0, 1)
            
        qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
        masks = torch.eye(num_qs, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
        qs_pred_grads = (1 - masks) * qs_pred_grads
        grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (num_qs - 1)
            
            #qfs_loss_total += self.eta * grad_loss
        qfs_loss_total += eta * grad_loss

    
    qfs_optimizer.zero_grad()
    qfs_loss_total.backward()
    qfs_optimizer.step()

    return qfs, target_qfs, [qfs_loss, grad_loss, qs_pred, q_target]


def torch_average(torch_list):
    out = torch.stack(torch_list)
    return torch.mean(out, axis = 0)

'''

def train_qfs(batch, trainer_list, alpha):
    batch = np_to_pytorch_batch(batch)

    discount=0.99  # Discount factor
    reward_scale=1.0  # Scaling of rewards to modulate entropy bonus
    qf_lr=0.0003  # Learning rate of Q functions
    optimizer_class=optim.Adam  # Class of optimizer for all networks
    soft_target_tau=5e-3  # Rate of update of target networks
    target_update_period=1 # How often to update target networks
    max_q_backup=False
    deterministic_backup=False
    num_qs=10
    qf_criterion = nn.MSELoss(reduction='none')
    eta = 1.0

    qfs = trainer_list[0].qfs
    qfs_optimizer = optimizer_class(
            qfs.parameters(),
            lr=qf_lr                    
        )

    obs= batch['observations']
    next_obs = batch['next_observations']
    actions = batch['actions']
    rewards = batch['rewards']
    terminals = batch['terminals']
        
    new_next_actions_list, new_log_pi_list, new_pi_list = [], [], []
        
    for i, agent in enumerate(trainer_list):
        new_next_actions, _, _, new_log_pi, *_ = agent.policy(
                next_obs,
                reparameterize=False,
                return_log_prob=True,
                )
        new_next_actions_list.append(new_next_actions)
        new_log_pi_list.append(new_log_pi)
        new_pi_list.append(torch.exp(new_log_pi))
        if i == len(trainer_list)-1:
            qs_pred = agent.qfs(obs, actions)
            new_next_actions = torch_average(new_next_actions_list)            
            target_q_values = agent.target_qfs.sample(next_obs, new_next_actions)

    # get log pi
    #new_log_pi = torch_average(new_log_pi_list)
    new_pi = torch_average(new_pi_list)
    new_log_pi = torch.log(new_pi)
    target_q_values -= alpha * new_log_pi

    future_values = (1. - terminals) * discount * target_q_values
    q_target = reward_scale * rewards + future_values
    qfs_loss = qf_criterion(qs_pred, q_target.detach().unsqueeze(0))
    qfs_loss = qfs_loss.mean(dim=(1, 2)).sum()

    qfs_loss_total = qfs_loss

    if eta > 0:
        obs_tile = obs.unsqueeze(0).repeat(num_qs, 1, 1)
        actions_tile = actions.unsqueeze(0).repeat(num_qs, 1, 1).requires_grad_(True)
        qs_preds_tile = qfs(obs_tile, actions_tile)
        qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
        qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
        qs_pred_grads = qs_pred_grads.transpose(0, 1)
            
        qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
        masks = torch.eye(num_qs, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
        qs_pred_grads = (1 - masks) * qs_pred_grads
        grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (num_qs - 1)
            
            #qfs_loss_total += self.eta * grad_loss
        qfs_loss_total += eta * grad_loss

    
    qfs_optimizer.zero_grad()
    qfs_loss_total.backward()
    qfs_optimizer.step()
    
    for agent in trainer_list:
        agent.qfs = qfs
        agent.target_qfs = target_qfs

    return qfs, target_qfs, [qfs_loss, grad_loss, qs_pred, q_target]

'''