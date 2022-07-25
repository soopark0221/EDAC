from experiment_utils.launch_experiment_ma import launch_experiment
from experiment_configs.configs.q_learning.sac_config import get_config
from experiment_configs.algorithms.offpolicy import get_offpolicy_algorithm

import argparse
import os 
import datetime
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def main(args):
    # Default parameters
    variant = dict(
        algorithm='SAC_sharedQ', #'SAC_buffer_3agents_maxminmax_gym_expl1000_buffer_1M',
        collector_type='step',
        env_name='hopper-random-v2',
        env_kwargs=dict(),
        replay_buffer_size=args.buffer_size, #int(1e6),
        reward_mean=False,  # added for easy config checking
        reward_std=-1.0,  # added for easy config checking
        policy_kwargs=dict(
            layer_size=256,
            num_q_layers=3,
            num_p_layers=3,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            use_automatic_entropy_tuning=True,
            policy_eval_start=0,
            num_qs=10,
            target_update_period=1,
            max_q_backup=False,
            deterministic_backup=False,
            eta=-1.0,
        ),
        offline_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=1000, #1000
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            save_snapshot_freq=3000, # save last epoch
        ),
    )

    experiment_kwargs = dict(
        exp_postfix='',
        use_gpu=True,
        log_to_tensorboard=False,
    )

    # Variant
    variant['env_name'] = args.env_name
    variant['seed'] = args.seed

    variant['offline_kwargs']['num_epochs'] = args.epoch

    # SAC-N
    variant['trainer_kwargs']['policy_lr'] = args.plr
    variant['trainer_kwargs']['qf_lr'] = args.qlr

    variant['trainer_kwargs']['num_qs'] = args.num_qs
    variant['trainer_kwargs']['max_q_backup'] = args.max_q_backup
    variant['trainer_kwargs']['deterministic_backup'] = args.deterministic_backup

    variant['reward_mean'] = args.reward_mean
    variant['reward_std'] = args.reward_std
    
    # EDAC
    variant['trainer_kwargs']['eta'] = args.eta

    # MA
    #variant['offline']=args.offline
    variant["offline_fraction"]= args.offline
    variant['num_agents']=args.num_agents

    # experiment name
    experiment_kwargs['exp_postfix'] = ''
    
    exp_postfix = '_{}qfs'.format(args.num_qs)
    exp_postfix= '_offline{}%'.format(int(args.offline*100))

    #learning rate 
    #exp_postfix += '_plr{:.4f}_qlr{:.4f}'.format(args.plr, args.qlr)
    exp_postfix += f'_{args.num_agents}agents'
    # exp_postfix += f'_buffer{variant["replay_buffer_size"]} # not that crucial info?
    exp_postfix += f'_offline{args.offline}'    # rather
    if variant['trainer_kwargs']['max_q_backup']:
        exp_postfix += '_maxq'
    if variant['trainer_kwargs']['deterministic_backup']:
        exp_postfix += '_detq'
    if args.eta > 0:
        exp_postfix += '_eta{:.2f}'.format(args.eta)
    if args.reward_mean:
        exp_postfix += '_mean'
    if args.reward_std > 0:
        exp_postfix += '_std'
    today=datetime.datetime.today()
    exp_postfix+=f'_time_{today.month}_{today.day}_{today.hour+9}:{today.minute}' # to avoid overwriting, time=KST

    experiment_kwargs['exp_postfix'] = exp_postfix

    experiment_kwargs['data_args'] = {
        'reward_mean': args.reward_mean,
        'reward_std': args.reward_std,
    }

    # Launch experiment
    launch_experiment(variant=variant,
                      get_config=get_config,
                      get_offpolicy_algorithm=get_offpolicy_algorithm,
                      **experiment_kwargs)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Variant
    parser.add_argument('-e',
                        '--env_name',
                        default='halfcheetah-random-v2',
                        type=str)
    parser.add_argument('--seed', default=0, type=int)
    # Misc arguments
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--log_to_tensorboard', action='store_true')
    parser.add_argument("--epoch", default=3000, type=int)
    # SAC
    parser.add_argument("--plr",
                        default=3e-4,
                        type=float,
                        help='policy learning rate')
    parser.add_argument("--qlr",
                        default=3e-4,
                        type=float,
                        help='Q learning rate')
    parser.add_argument("--num_qs",
                        default=10,
                        type=int,
                        help='number of Q-functions to be used')
    parser.add_argument('--max_q_backup',
                        action='store_true',
                        help='use max q backup')
    parser.add_argument('--deterministic_backup',
                        action='store_true',
                        help='use deterministic backup')
    parser.add_argument('--eta',
                        default=-1.0,
                        type=float,
                        help='eta for diversifying Q-ensemble. < 0 for SAC-N.')
    
    # reward preprocessing
    parser.add_argument("--reward_mean",
                        action='store_true',
                        help='normalize rewards to 0 mean')
    parser.add_argument("--reward_std",
                        action='store_true',
                        help='normalize rewards to 1 std')
    #agents
    parser.add_argument("--num_agents",
                        default=2,
                        type=int,
                        help='number of agents')
    # offline dataset (d4rl)
    #arser.add_argument("--offline",
    #                    default=True,
    #                    help='use offline data')
    parser.add_argument("--offline",
                        default=1.0,
                        type=float,
                        help='fraction of using offline data')
    # replay buffer
    parser.add_argument("--buffer_size",
                        default=int(1e6),
                        type=int,
                        help='buffer_size')
    
    args = parser.parse_args()

    main(args)