import numpy as np
from matrpo.trainer.plot import plot
from matrpo.trainer.matrpo import MATRPO
from baselines.bench.monitor import load_results

seed = 1
mode = 'trpo'
scenario = 'scenario'
network_kwargs = {'num_layers': 2, 'num_hidden': 128, 'activation': 'selu'}
agents = MATRPO(
    env_id='Microgrid-v2',
    scenario_id=scenario,
    nsteps=24*10,
    network='mlp',
    num_env=1,
    seed=seed,
    finite=True,
    admm_iter=100,
    logger_dir=None,
    load_path='/home/lihepeng/Documents/Github/matrpo/results/graphs/matrpo/s{}'.format(seed),
    info_keywords=tuple('r{}'.format(i) for i in range(10)),
    mode=mode,
    **network_kwargs)

agents.model.test = True
env = agents.test_env.unwrapped
obs_n = env.reset(day=1)
            while True:
                # query for action from each agent's policy
                act_n, _, _ = agents.model.step(obs_n)
                # step environment
                obs_n, reward_n, done_n, info_n = agents.test_env.unwrapped.step(act_n)
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
                    breakinfo_