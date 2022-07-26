import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from collections import OrderedDict

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.torch.distributions import TanhNormal
from lifelong_rl.util.eval_util import create_stats_ordered_dict
from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from lifelong_rl.torch.pytorch_util import np_to_pytorch_batch

# from torch_batch_svd import svd

ACTION_MIN = -1.0
ACTION_MAX = 1.0


class SharedQTrainer(TorchTrainer):
    """
    Soft Actor Critic (Haarnoja et al. 2018). (Offline training ver.)
    Continuous maximum Q-learning algorithm with parameterized actor.
    """
    def __init__(
            self,
            env,  # Associated environment for learning
            policy,  # Associated policy (should be TanhGaussian)
            qfs,  # Q functions
            target_qfs,  # Slow updater to Q functions
            discount=0.99,  # Discount factor
            use_automatic_entropy_tuning=True,  # Whether to use the entropy-constrained variant
            target_entropy=None,  # Target entropy for entropy-constraint variant
            policy_lr=3e-4,  # Learning rate of policy and entropy weight
            optimizer_class=optim.Adam,  # Class of optimizer for all networks
            deterministic_backup=False,
            replay_buffer=None,
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs

        self.discount = discount

        self.deterministic_backup = deterministic_backup

        self.replay_buffer = replay_buffer
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # Heuristic value: dimension of action space
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat,
                                           1).view(obs.shape[0] * num_repeat,
                                                   obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(-1, obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions,
                                           1).view(obs.shape[0] * num_actions,
                                                   obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp,
            reparameterize=True,
            return_log_prob=True,
        )
        return new_obs_actions.detach(), new_obs_log_pi.view(
            obs.shape[0], num_actions, 1).detach()

    def train_from_torch(self, batch, indices, eta, Qmin=True):
        obs= batch['observations']
        next_obs = batch['next_observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
    
        if eta > 0:
            actions.requires_grad_(True)
        
        """
        Policy and Alpha Loss
        """

        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        
        if Qmin == True:
            q_new_actions = self.qfs.sample(obs, new_obs_actions)
        else: 
            q_new_actions = self.qfs.sample_max(obs, new_obs_actions)

        policy_loss = (alpha * log_pi - q_new_actions).mean()
            
        if self.use_automatic_entropy_tuning and not self.deterministic_backup:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            policy_loss = ptu.get_numpy(log_pi - q_new_actions).mean()
            policy_avg_std = ptu.get_numpy(torch.exp(policy_log_std)).mean()
            self.eval_statistics['Policy Loss'] = np.mean(policy_loss)

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
            self.eval_statistics['Policy std'] = np.mean(policy_avg_std)

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qfs,
            self.target_qfs,
        ]

        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.qfs,
            log_alpha=self.log_alpha,
            policy_optim=self.policy_optimizer,
            alpha_optim=self.alpha_optimizer,
        )
