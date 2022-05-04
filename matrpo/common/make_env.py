import os, gym
import multiagent
import multiagent.scenarios as scenarios

from matrpo.common.monitor import Monitor
from matrpo.common.vec_env.subproc_vec_env import SubprocVecEnv

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

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

def make_env(env_id, scenario_id, seed, logger_dir, reward_scale, info_keywords, mpi_rank=0, subrank=0):
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
    env = gym.make(env_id, **env_dict)
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

def make_vec_env(env_id, scenario_id, seed, num_env, logger_dir, reward_scale=1.0, force_dummy=False, info_keywords=()):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id, 
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
