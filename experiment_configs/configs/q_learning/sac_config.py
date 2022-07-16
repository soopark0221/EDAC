from lifelong_rl.models.networks import ParallelizedEnsembleFlattenMLP
#from lifelong_rl.policies.base.base import MakeDeterministic
from lifelong_rl.policies.models.tanh_gaussian_policy import TanhGaussianPolicy
from lifelong_rl.trainers.q_learning.sac import SACTrainer
import lifelong_rl.util.pythonplusplus as ppp
import os
import torch
import lifelong_rl.torch.pytorch_util as ptu
from torch.nn import functional as F

import os
import torch.nn as nn
import abc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"


class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass


class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass


class MakeDeterministic(nn.Module, Policy):
    def __init__(self, stochastic_policy):
        super().__init__()
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
):
    """
    Policy construction
    """

    num_qs = variant['trainer_kwargs']['num_qs']
    M = variant['policy_kwargs']['layer_size']
    num_q_layers = variant['policy_kwargs']['num_q_layers']
    num_p_layers = variant['policy_kwargs']['num_p_layers']

    num_agents = variant['num_agents']  # XXX
    policy_list = []
    trainer_list =[]
    eval_policy_list = []

    qfs, target_qfs = ppp.group_init(
        2,
        ParallelizedEnsembleFlattenMLP,
        ensemble_size=num_qs,
        hidden_sizes=[M] * num_q_layers,
        input_size=obs_dim + action_dim,
        output_size=1,
        layer_norm=None,
        )
    for i in range(num_agents):
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M] * num_p_layers,
            layer_norm=None,
        )
        policy_list.append(policy)

        trainer = SACTrainer(
            env=eval_env,
            policy=policy,
            qfs=qfs,
            target_qfs=target_qfs,
            replay_buffer=replay_buffer,
            **variant['trainer_kwargs'],
        )
        trainer_list.append(trainer)
        eval_policy = MakeDeterministic(policy)
        eval_policy_list.append(eval_policy)

    

    """
    Create config dict
    """

    config = dict()
    config.update(
        dict(
            trainer_list=trainer_list,  # XXX trainer_list
            exploration_policy_list=policy_list,  # XXX policy_list
            evaluation_policy_list=eval_policy_list,  # XXX policy_list
            exploration_env=expl_env,
            evaluation_env=eval_env,
            replay_buffer=replay_buffer,  # XXX double check: using the same replay_buffer
        ))
    config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())

    return config
