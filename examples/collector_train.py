import os, gym
import numpy as np
from scipy.linalg import toeplitz
from baselines.bench.monitor import load_results
from matrpo.common.make_env import make_env, make_vec_env
from matrpo.trainer.plot import plot
from matrpo.trainer.matrpo import MATRPO

if __name__ == '__main__':
    num_agents = 8
    comm_matrix = toeplitz(
        [1]+[0]*(num_agents-2), 
        [1,-1]+[0]*(num_agents-2)
    ).astype(np.float32)
    adj_matrix = np.vstack([
        comm_matrix,
        np.array([[-1]+[0]*(num_agents-2)+[1]]),
    ]).astype(np.float32)

    seed = 1
    num_env = 10
    mode = 'trpo'
    env_id = 'MultiAgent-v0'
    scenario_id = 'collector'
    logger_dir = '/home/lihepeng/Documents/Github/matrpo/results/rewards/collector'
    info_keywords = tuple('r{}'.format(i) for i in range(8))
    env = make_vec_env(env_id, scenario_id, seed, num_env, logger_dir, reward_scale=1.0, info_keywords=info_keywords)
    test_env = make_env(env_id, scenario_id, seed, logger_dir, reward_scale=1.0, info_keywords=info_keywords)
    network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'selu'}
    agents = MATRPO(
        env=env,
        adj_matrix=adj_matrix,
        nsteps=1000,
        network='mlp',
        seed=seed,
        finite=True,
        admm_iter=500,
        num_env=num_env, 
        load_path='/home/lihepeng/Documents/Github/matrpo/results/graphs/collector',
        mode=mode,
        test_env=test_env,
        **network_kwargs)
    # training
    total_timesteps = 10000
    for step in range(1, total_timesteps+1):
        actions, obs, rewards, returns, dones, values, advs, neglogpacs = agents.runner.run()
        agents.model.train(actions, obs, rewards, returns, dones, values, advs, neglogpacs)
        df_train = load_results('/home/lihepeng/Documents/Github/matrpo/results/rewards/collector')
        if step % 10 == 1:
            plot(df_train, agents, 100)
            agents.model.save()
            # agents.play()

