import gtimer as gt

import abc

from lifelong_rl.core import logger
from lifelong_rl.core.rl_algorithms.rl_algorithm import _get_epoch_timings
from lifelong_rl.util import eval_util


class OffpolicyRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
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
            min_num_steps_before_training,
            num_eval_steps_per_epoch,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            save_snapshot_freq=1000,
    ):
        self.trainer_list = trainer_list  # list
        self.eval_policy_list = evaluation_policy_list  # list
        self.eval_env = evaluation_env
        self.expl_env = exploration_env
        self.eval_data_collector_list = evaluation_data_collector_list  # list
        self.expl_data_collector_list = exploration_data_collector_list  # list

        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.replay_buffer = replay_buffer  # check 

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.save_snapshot_freq = save_snapshot_freq

        self._start_epoch = 0
        self.post_epoch_funcs = []

    def _train(self):
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            '''
            if epoch == 0 and self.min_num_steps_before_training > 0:
                init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                )
                self.replay_buffer.add_paths(init_expl_paths)
                self.expl_data_collector.end_epoch(-1) 
            self.eval_data_collector.collect_new_paths(
                max_path_length=self.max_path_length,
                num_samples=self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                alpha=curr_alpha,
            )
            '''
            for agent in self.trainer_list:  # TODO : check the first one
                if hasattr(agent, 'log_alpha'):
                    curr_alpha = agent.log_alpha.exp()
                else:
                    curr_alpha = None

            gt.stamp('evaluation sampling')
            eta = 1.0*(0.7**(epoch / self.num_epochs))
            for i, agent in enumerate(self.trainer_list):
                self.eval_data_collector_list[i].collect_new_paths(
                max_path_length=self.max_path_length,
                num_samples=self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                alpha=curr_alpha,
                )
                #Qmin = True if i//2==1 else False # max-min-max-min
                Qmin = True if i//2==0 else False # min-max-min
                #Qmin = False
                for _ in range(self.num_train_loops_per_epoch):
                    new_expl_paths = self.expl_data_collector_list[i].collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)
                    
                    gt.stamp('data storing', unique=False)
                    self.training_mode(True)
                    for _ in range(self.num_trains_per_train_loop):
                        train_data, indices = self.replay_buffer.random_batch(
                            self.batch_size, return_indices=True)
                        agent.train(train_data, indices, Qmin=Qmin, eta=eta)
                    self.training_mode(False)
                self._end_epoch(epoch, agent, self.eval_data_collector_list[i], self.expl_data_collector_list[i])
            gt.stamp('training')

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _end_epoch(self, epoch, agent, eval_data_collector, expl_data_collector):
        snapshot = self._get_snapshot(agent)
        for k, v in snapshot.items():
            if self.save_snapshot_freq is not None and \
                    (epoch + 1) % self.save_snapshot_freq == 0:
                logger.save_itr_params(epoch + 1, snapshot[k], prefix='offline_itr')
            gt.stamp('saving', unique=False)

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
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)
        """
        Replay Buffer
        """
        logger.record_dict(self.replay_buffer.get_diagnostics(),
                           prefix='replay_buffer/')
        """
        Trainer
        """
        training_diagnostics = self._get_training_diagnostics_dict(agent)
        print(f'training diag {training_diagnostics.keys}')
        for prefix in training_diagnostics:
            print(prefix)
            logger.record_dict(training_diagnostics[prefix],
                               prefix=prefix + '/')

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
        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
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
