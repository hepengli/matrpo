import numpy as np
import matplotlib.pyplot as plt

import gym

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.bench.monitor import load_results
from baselines.common import retro_wrappers, set_global_seeds

from matrpo.common.monitor import Monitor
from matrpo.common.vec_env.subproc_vec_env import SubprocVecEnv
from matrpo.trainer.model import Model
from matrpo.trainer.runner import Runner
from matrpo.trainer.build_policy import Policy
import multiagent
import multiagent.scenarios as scenarios

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        # action = action.numpy()
        # # unstack action spaces
        # action_spaces = []
        # if isinstance(self.env.unwrapped.action_space, gym.spaces.Tuple):
        #     action_spaces.extend(list(self.env.unwrapped.action_space.spaces))
        # else:
        #     action_spaces.append(self.env.unwrapped.action_space)
        # # clip actions
        # s, t = 0, 0
        # for space in action_spaces:
        #     if isinstance(space, gym.spaces.Box):
        #         t += space.shape[0]
        #         clipped_action = np.clip(action[s:t], -3, 3) # normalize the action
        #         action[s:t] = ((clipped_action + 3) / 6) * (space.high - space.low) + space.low
        #     elif isinstance(space, gym.spaces.Discrete):
        #         t += 1
        #     elif isinstance(space, gym.spaces.MultiDiscrete):
        #         t += len(space.nvec)
        #     s = t
        # action = np.nan_to_num(action)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class MATRPO(object):
    """ Paralell CPO algorithm """
    def __init__(self, env_id, scenario_id, nsteps, network, num_env, admm_iter, seed=None, 
                 finite=True, load_path=None, logger_dir=None, force_dummy=False, mode='matrpo', 
                 gamma=0.995, lam=0.95, max_kl=0.001, ent_coef=0.0, vf_stepsize=3e-4, vf_iters=3, 
                 cg_damping=1e-2, cg_iters=10, lbfgs_iters=10, rho=1.0, reward_scale=1.0, 
                 ob_normalization=False, info_keywords=(), **network_kwargs):

        set_global_seeds(seed)
        np.set_printoptions(precision=5)
        self.env_id = env_id

        # Scenario
        scenario = scenarios.load('{}.py'.format(scenario_id)).Scenario()
        world = scenario.make_world()
        nbatch = num_env * nsteps

        # Environment
        self.env = env = self.make_vec_env(
            scenario_id, seed, num_env, logger_dir, reward_scale, force_dummy, info_keywords)
        self.test_env = self.make_env(
            scenario_id, seed, logger_dir, reward_scale, info_keywords)

        # create interactive policies for each agent
        self.policies = policies = Policy(env, world, network, nbatch, mode, rho, max_kl, ent_coef, 
                vf_stepsize, vf_iters, cg_damping, cg_iters, lbfgs_iters, load_path, **network_kwargs)

        # model
        self.model = model = Model(env, world, policies, admm_iter, mode, ob_normalization)

        # runner
        if num_env > 1:
            self.runner = Runner(env, world, model, nsteps, gamma, lam, finite)

    def make_env(self, scenario_id, seed, logger_dir, reward_scale, info_keywords, mpi_rank=0, subrank=0):
        """
        Create a wrapped, monitored gym.Env for safety.
        """
        scenario = scenarios.load('{}.py'.format(scenario_id)).Scenario()
        if not hasattr(scenario, 'post_step'): scenario.post_step = None
        world = scenario.make_world()
        env_dict = {
            "world": world,
            'reset_callback': scenario.reset_world,
            'reward_callback': scenario.reward, 
            'observation_callback': scenario.observation,
            'post_step_callback': scenario.post_step,
            'info_callback': scenario.benchmark_data,
            'shared_viewer':  True,
            }
        env = gym.make(self.env_id, **env_dict)
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

    def make_vec_env(self, scenario_id, seed, num_env, logger_dir, reward_scale, force_dummy, info_keywords):
        """
        Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
        """
        mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        seed = seed + 10000 * mpi_rank if seed is not None else None
        def make_thunk(rank, initializer=None):
            return lambda: self.make_env(
                scenario_id,
                seed,
                logger_dir,
                reward_scale,
                info_keywords,
                mpi_rank=mpi_rank,
                subrank=rank)

        if not force_dummy and num_env > 1:
            return SubprocVecEnv([make_thunk(i) for i in range(num_env)])
        else:
            return make_thunk(0)()

    def play(self):
        self.model.test = True
        for _ in range(10):
            obs_n = self.test_env.reset()
            while True:
                # query for action from each agent's policy
                act_n, _, _ = self.model.step(obs_n)
                # step environment
                obs_n, reward_n, done_n, _ = self.test_env.step(act_n)
                # render all agent views
                self.test_env.render()
                import time
                time.sleep(0.05)
                # display rewards
                for i, agent in enumerate(self.test_env.unwrapped.world.agents):
                    print(agent.name + " reward: %0.3f" % reward_n[i])
                # break
                if done_n:
                    print('done!')
                    time.sleep(2)
                    break

        self.model.test = False