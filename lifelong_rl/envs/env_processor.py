import gym
import d4rl

from lifelong_rl.envs.wrappers import NormalizedBoxEnv, NonTerminatingEnv, SwapColorEnv


gym.logger.set_level(40)  # stop annoying Box bound precision error


def make_env(env_name, terminates=True, offline=0.0, **kwargs):
    env = None
    env_infos = dict()

    """
    Offline reinforcement learning w/ d4rl
    TODO: set env_infos['mujoco']=False for non-mujoco tasks
    """
    if offline > 0:
        print('Using d4rl')
        env = gym.make(env_name)
        if any(phrase in env_name for phrase in ['halfcheetah', 'hopper', 'walker2d', 'antmaze']):
            env_infos['mujoco'] = True
        else:
            env_infos['mujoco'] = False
    else:
        print('Not using d4rl')
        env = gym.make(env_name)
        if any(phrase in env_name for phrase in ['HalfCheetah', 'Hopper', 'Walker2d', 'Ant']):
            env_infos['mujoco'] = True
        else:
            env_infos['mujoco'] = False

    if not terminates:
        env = NonTerminatingEnv(env)
    
    
    return env, env_infos
