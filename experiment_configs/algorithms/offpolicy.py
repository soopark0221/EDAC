from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchOffpolicyRLAlgorithm


def get_offpolicy_algorithm(config, expl_path_collector, eval_path_collector):

    algorithm = TorchOffpolicyRLAlgorithm(
        trainer=config['trainer'],
        exploration_policy=config['exploration_policy'],
        evaluation_policy=config['evaluation_policy'],
        evaluation_env=config['evaluation_env'],
        replay_buffer=config['replay_buffer'],
        evaluation_data_collector=eval_path_collector,
        exploration_data_collector=expl_path_collector,
        **config['offline_kwargs']
    )

    return algorithm
