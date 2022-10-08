import os, gym
import numpy as np
from scipy.linalg import toeplitz
from matrpo.trainer.plot import plot
from matrpo.trainer.matrpo import MATRPO

import multiagent
import multiagent.scenarios as scenarios

from baselines.bench.monitor import load_results
from baselines.common import retro_wrappers

from matrpo.common.monitor import Monitor
from matrpo.common.vec_env.subproc_vec_env import SubprocVecEnv

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from smac.env import StarCraft2Env
import numpy as np
import gym
import gym.spaces as spaces

class SC2Env(StarCraft2Env, gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }
    def __init__(self, map_name, seed=None, **kwargs):
        super().__init__(map_name=map_name, seed=seed, **kwargs)
        env_info = self.get_env_info()
        n_actions = env_info['n_actions']
        obs_dim = env_info['obs_shape']
        self.n_agents = n_agents = env_info['n_agents']
        self.seed(seed)

        self.action_space = []
        self.observation_space = []
        for agent_id in range(n_agents):
            self.action_space.append(spaces.Discrete(n_actions))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

    def seed(self, seed):
        self._seed = super().seed()
        return self._seed

    def reset(self):
        obs_n, state = super().reset()
        return obs_n

    def step(self, actions):
        for agent_id, action in enumerate(actions):
            avail_actions = self.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            if action not in avail_actions_ind:
                action = np.random.choice(avail_actions_ind)
            actions[agent_id] = action
        reward, terminated, info = super().step(actions)
        obs_n = self.get_obs()
        reward_n = [reward] * self.n_agents
        done_n = [terminated] * self.n_agents
        info_n = {}
        info_n.update(dict(zip(['r{}'.format(i) for i in range(10)], reward_n)))
        info_n.update(info)

        return obs_n, reward_n, done_n, info_n

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def make_env(map_name, seed=None, logger_dir=None, reward_scale=1.0, info_keywords=(), mpi_rank=0, subrank=0, **kwargs):
    """
    Create a wrapped, monitored gym.Env for safety.
    """
    env = SC2Env(map_name, seed, **kwargs)
    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                allow_early_resets=True,
                info_keywords=info_keywords)
    env = ClipActionsWrapper(env)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

def make_vec_env(map_name, seed, num_env, logger_dir, reward_scale=1.0, force_dummy=False, info_keywords=(), **kwargs):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            map_name=map_name,
            seed=seed,
            logger_dir=logger_dir,
            reward_scale=reward_scale,
            info_keywords=info_keywords,
            mpi_rank=mpi_rank,
            subrank=rank,
            **kwargs)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i) for i in range(num_env)])
    else:
        return make_thunk(0)()


def main():
    num_agents = 10
    comm_matrix = toeplitz(
        [1]+[0]*(num_agents-1), 
        [1,-1]+[0]*(num_agents-1)
    ).astype(np.float32)
    adj_matrix = np.vstack([
        comm_matrix,
        np.array([[-1]+[0]*(num_agents-2)+[1]]),
    ]).astype(np.float32)

    seed = 1
    num_env = 10
    mode = 'trpo'
    map_name = '3s5z_vs_3s6z'
    logger_dir = '/home/lihepeng/Documents/Github/matrpo/results/rewards/{}'.format(map_name)
    test_dir = '/home/lihepeng/Documents/Github/matrpo/results/test/{}'.format(map_name)
    load_path = '/home/lihepeng/Documents/Github/matrpo/results/graphs/{}'.format(map_name)
    replay_path = '/home/lihepeng/Documents/Github/matrpo/results/replays/{}'.format(map_name)
    info_keywords = tuple('r{}'.format(i) for i in range(10)) + ('battle_won', 'dead_allies', 'dead_enemies')
    env = make_vec_env(map_name, seed, num_env, logger_dir, reward_scale=1.0, info_keywords=info_keywords)
    test_args = {'replay_dir':replay_path, 'replay_prefix':'{}'.format(map_name)}
    test_env = make_env(map_name, seed, test_dir, info_keywords=info_keywords, **test_args)
    network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'selu'}
    agents = MATRPO(
        env=env,
        test_env=test_env,
        adj_matrix=adj_matrix,
        nsteps=1000,
        max_kl=0.001,
        network='mlp',
        seed=seed,
        admm_iter=500,
        num_env=num_env, 
        load_path=load_path,
        mode=mode,
        **network_kwargs)
    # training

    total_timesteps = 1001
    for step in range(total_timesteps+1):
        actions, obs, rewards, returns, dones, values, advs, neglogpacs = agents.runner.run()
        agents.model.train(actions, obs, rewards, returns, dones, values, advs, neglogpacs)
        if step % 1 == 0:
            # agents.evaluate(32)
            df_train = load_results(logger_dir)
            # df_test = load_results(test_dir)
            plot(df_train, agents, 100)
            agents.model.save()
            stats = df_train[list(info_keywords)[-3:]].iloc[-100:].mean(axis=0).to_dict()
            # test_stats = df_test[list(info_keywords)[-3:]].iloc[-32:].mean(axis=0).to_dict()
            # test_stats = dict([('test_'+k, v) for (k, v) in test_stats.items()])
            log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(step, df_train.shape[0])
            for (k, v) in sorted(stats.items()):
                log_str += "{:<25}{:>8}".format(k + ":", v)
                log_str += "\t"
            # log_str += "\n"
            # for (k, v) in sorted(test_stats.items()):
            #     log_str += "{:<25}{:>8}".format(k + ":", v)
            #     log_str += "\t"
            print(log_str)

if __name__ == '__main__':
    main()


