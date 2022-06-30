import gtimer as gt

import abc

from lifelong_rl.core import logger
from lifelong_rl.core.rl_algorithms.rl_algorithm import _get_epoch_timings
from lifelong_rl.util import eval_util


class OffpolicyRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_policy,
            evaluation_policy,
            evaluation_env,
            evaluation_data_collector,
            exploration_data_collector,
            replay_buffer,
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
        self.trainers = []
        self.trainer = trainer
        self.eval_policy = evaluation_policy
        self.eval_env = evaluation_env
        self.eval_data_collectors = []
        self.eval_data_collector = evaluation_data_collector        
        self.expl_data_collectors = []
        self.expl_data_collector = exploration_data_collector
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.replay_buffer = replay_buffer

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.save_snapshot_freq = save_snapshot_freq

        self._start_epoch = 0
        self.post_epoch_funcs = []

        self.agent_n = 2
        for i in range(self.agent_n):
            self.trainers.append(self.trainer)
            self.eval_data_collectors.append(self.eval_data_collector)
            self.expl_data_collectors.append(self.expl_data_collector)
    def _train(self):
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            #if epoch == 0 and self.min_num_steps_before_training > 0:
            #    init_expl_paths = self.expl_data_collector.collect_new_paths(
            #    self.max_path_length,
            #    self.min_num_steps_before_training,
            #    discard_incomplete_paths=False,
            #    )
            #    self.replay_buffer.add_paths(init_expl_paths)
            #    self.expl_data_collector.end_epoch(-1) 
            if hasattr(self.trainer, 'log_alpha'):
                curr_alpha = self.trainer.log_alpha.exp()
            else:
                curr_alpha = None
            #self.eval_data_collector.collect_new_paths(
            #    max_path_length=self.max_path_length,
            #    num_samples=self.num_eval_steps_per_epoch,
            #    discard_incomplete_paths=True,
            #    alpha=curr_alpha,
            #)
            gt.stamp('evaluation sampling')

            for i, agent in enumerate(self.trainers):
                self.eval_data_collectors[i].collect_new_paths(
                max_path_length=self.max_path_length,
                num_samples=self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                alpha=curr_alpha,
                )
                for _ in range(self.num_train_loops_per_epoch):
                    new_expl_paths = self.expl_data_collectors[i].collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)
                    
                    #print(f' replay buffer {self.replay_buffer}')
                    #print(f' new expl path is {new_expl_paths}')
                    self.replay_buffer.add_paths(new_expl_paths)
                    gt.stamp('data storing', unique=False)
                    
                    self.training_mode(True)
                    for _ in range(self.num_trains_per_train_loop):
                        train_data, indices = self.replay_buffer.random_batch(
                            self.batch_size, return_indices=True)
                        #self.trainer.train(train_data, indices)
                        agent.train(train_data, indices)


                self._end_epoch(epoch, agent, self.eval_data_collectors[i], self.expl_data_collectors[i])
            self.training_mode(False)
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

        if hasattr(self.eval_policy, 'end_epoch'):
            self.eval_policy.end_epoch(epoch)

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
