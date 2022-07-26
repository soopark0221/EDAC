import gtimer as gt

import abc

from lifelong_rl.core import logger
from lifelong_rl.core.rl_algorithms.rl_algorithm import _get_epoch_timings
from lifelong_rl.util import eval_util
from lifelong_rl.torch.pytorch_util import np_to_pytorch_batch
import lifelong_rl.torch.pytorch_util as ptu
import torch
import torch.optim as optim
from torch import nn as nn


class OffpolicyRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            qfs,
            target_qfs,
            trainer_list,  # list
            exploration_policy_list,  # list
            evaluation_policy_list,  # list
            evaluation_env,
            exploration_env, 
            evaluation_data_collector_list,  # list
            exploration_data_collector_list,  # list
            replay_buffer,  # TODO
            batch_size,
            max_path_length,
            num_epochs,
            num_expl_steps_per_train_loop,
            num_eval_steps_per_epoch,
            num_trains_per_train_loop,
            num_qs=10,
            qf_lr=3e-4,
            discount=0.99,
            eta=-1.0,
            soft_target_tau=5e-3,
            target_update_period=1,
            save_snapshot_freq=1000,
            deterministic_backup=False,
            reward_scale=1.0,
            optimizer_class=optim.Adam
    ):  
        # initialize qfs training
        self.qfs=qfs
        self.target_qfs=target_qfs

        self.num_qs=num_qs
        self.qf_lr=qf_lr
        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        
        self.deterministic_backup = deterministic_backup
        self.eta=eta

        self.qf_criterion = nn.MSELoss(reduction='none')
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr,
        )

        # Algorithm
        self.trainer_list = trainer_list  # list
        self.eval_policy_list = evaluation_policy_list  # list
        self.eval_env = evaluation_env
        self.expl_env = exploration_env
        self.eval_data_collector_list = evaluation_data_collector_list  # list
        self.expl_data_collector_list = exploration_data_collector_list  # list

        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop

        self.replay_buffer = replay_buffer  # check 

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.save_snapshot_freq = save_snapshot_freq

        self._start_epoch = 0
        self.post_epoch_funcs = []

    def _train(self):
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
                ):

            if self.eta>0:    
                self.eta*=(0.7**(epoch / self.num_epochs))
                
            for i, agent in enumerate(self.trainer_list):
                if hasattr(agent, 'log_alpha'):
                    curr_alpha = agent.log_alpha.exp()
                else:
                    curr_alpha = 1.0

                self.eval_data_collector_list[i].collect_new_paths(
                max_path_length=self.max_path_length,
                num_samples=self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                alpha=curr_alpha,
                )
                gt.stamp('evaluation sampling',unique=False)
                new_expl_paths = self.expl_data_collector_list[i].collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling',unique=False)
                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing',unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data, indices = self.replay_buffer.random_batch(
                    self.batch_size, return_indices=True)
                for i,agent in enumerate(self.trainer_list):
                    Qmin = True if i%2==0 else False # min-max-min
                    agent.train(train_data, indices, self.eta, Qmin=Qmin)
                self.train_qfs(train_data)
            self.training_mode(False)
            gt.stamp('training',unique=False)
            for i,agent in enumerate(self.trainer_list):
                self._end_epoch(epoch, agent, self.eval_data_collector_list[i], self.expl_data_collector_list[i])

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def train_qfs(self, batch):
        batch=np_to_pytorch_batch(batch)
        obs= batch['observations']
        next_obs = batch['next_observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']

        if self.eta>0:
            actions.requires_grad_(True)
        
        qs_pred = self.qfs(obs, actions)
        new_next_actions_list=[]
        alpha_list=[]
        for agent in self.trainer_list:
            new_next_actions, _, _, _, *_ = agent.policy(
                next_obs,
                reparameterize=False,
                return_log_prob=True,
            )
            new_next_actions_list.append(new_next_actions)
            if hasattr(agent, 'log_alpha'):
                alpha = agent.log_alpha.exp()
            else:
                alpha = 1.0
            alpha_list.append(alpha)
        avg_next_actions=self.torch_average(new_next_actions_list)
        avg_alpha=self.torch_average(alpha_list)

        target_q_values = self.target_qfs.sample(next_obs, avg_next_actions)
        if not self.deterministic_backup:
            new_log_pi_list=[]
            for agent in self.trainer_list:
                new_log_pi=agent.policy.get_log_probs(next_obs, avg_next_actions)
                new_log_pi_list.append(new_log_pi)
            avg_new_log_pis=self.torch_average(new_log_pi_list)
            target_q_values -= alpha * avg_new_log_pis

        future_values = (1. - terminals) * self.discount * target_q_values
        q_target = self.reward_scale * rewards + future_values
        qfs_loss = self.qf_criterion(qs_pred, q_target.detach().unsqueeze(0))
        qfs_loss = qfs_loss.mean(dim=(1, 2)).sum()

        qfs_loss_total = qfs_loss
        
        if self.eta > 0:
            obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
            actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
            qs_preds_tile = self.qfs(obs_tile, actions_tile)
            qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
            qs_pred_grads = qs_pred_grads.transpose(0, 1)
            
            qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
            masks = torch.eye(self.num_qs, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
            qs_pred_grads = (1 - masks) * qs_pred_grads
            grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (self.num_qs - 1)
            
            qfs_loss_total += self.eta * grad_loss

        self.qfs_optimizer.zero_grad()
        qfs_loss_total.backward()
        self.qfs_optimizer.step()

        train_steps=self.trainer_list[0]._num_train_steps
        self.try_update_target_networks(train_steps)

    def try_update_target_networks(self, train_steps):
        if train_steps % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.qfs, self.target_qfs,
                                self.soft_target_tau)

    def torch_average(self,torch_list):
        out = torch.stack(torch_list)
        return torch.mean(out, axis = 0)


    def _end_epoch(self, epoch, agent, eval_data_collector, expl_data_collector):

        snapshot = self._get_snapshot(agent)
        for k, v in snapshot.items():
            if self.save_snapshot_freq is not None and \
                    (epoch + 1) % self.save_snapshot_freq == 0:
                logger.save_itr_params(epoch + 1, snapshot[k], prefix='offline_itr')
            gt.stamp('saving',unique=False)

        self._log_stats(epoch, agent, eval_data_collector, expl_data_collector)

        self._end_epochs(epoch, agent, eval_data_collector, expl_data_collector)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self, agent):
        snapshot = {}

        for k, v in agent.get_snapshot().items():
            snapshot['trainer/'+ k] = v
        '''
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        '''
        return snapshot

    def _end_epochs(self, epoch, agent, eval_data_collector, expl_data_collector):
        eval_data_collector.end_epoch(epoch)
        expl_data_collector.end_epoch(epoch)
        agent.end_epoch(epoch)

        for e in self.eval_policy_list:
            if hasattr(e, 'end_epoch'):
                e.end_epoch(epoch)

    def _get_trainer_diagnostics(self, trainer):
        return trainer.get_diagnostics()

    def _get_training_diagnostics_dict(self, agent):
        diag_dict = {}

        diag_dict[f'policy_trainer'] = self._get_trainer_diagnostics(agent)
        #return {'policy_trainer': self._get_trainer_diagnostics()}
        return diag_dict

    def _log_stats(self, epoch, agent, eval_data_collector, expl_data_collector):
        logger.log("Epoch {} of agent {} finished".format(epoch, self.trainer_list.index(agent)), with_timestamp=True) #agent
        """
        Replay Buffer
        """
        logger.record_dict(self.replay_buffer.get_diagnostics(),
                           prefix='replay_buffer/')
                           
        """
        Exploration
        """
        logger.record_dict(
            expl_data_collector.get_diagnostics(),
            prefix='expl/'
        )
        expl_paths = expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='expl/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="expl/",
        )
        """
        Evaluation
        """
        if self.num_eval_steps_per_epoch > 0:
            logger.record_dict(
                eval_data_collector.get_diagnostics(),
                prefix='evaluation/',
            )
            eval_paths = eval_data_collector.get_epoch_paths()
            if hasattr(self.eval_env, 'get_diagnostics'):
                logger.record_dict(
                    self.eval_env.get_diagnostics(eval_paths),
                    prefix='evaluation/',
                )
            logger.record_dict(
                eval_util.get_generic_path_information(eval_paths),
                prefix="evaluation/",
            )
        """
        Misc
        """
        # time stamp logging early for csv format
        gt.stamp('logging',unique=False) # Changed
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Agent', self.trainer_list.index(agent))
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        #gt.stamp('logging', unique=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
