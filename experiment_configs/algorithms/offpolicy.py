from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchOffpolicyRLAlgorithm


def get_offpolicy_algorithm(config, expl_path_collector_list, eval_path_collector_list):

    algorithm = TorchOffpolicyRLAlgorithm(
        qfs=config['qfs'],
        target_qfs=config['target_qfs'],
        trainer_list=config['trainer_list'],
        exploration_policy_list=config['exploration_policy_list'],
        evaluation_policy_list=config['evaluation_policy_list'],
        evaluation_env=config['evaluation_env'],
        exploration_env=config['exploration_env'],
        replay_buffer=config['replay_buffer'],
        evaluation_data_collector_list=eval_path_collector_list,
        exploration_data_collector_list=expl_path_collector_list,
        **config['offline_kwargs']
    )

    return algorithm
