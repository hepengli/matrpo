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
    num_env=8,
    seed=seed,
    finite=True,
    admm_iter=100,
    logger_dir='/home/lihepeng/Documents/Github/matrpo/results/rewards/matrpo/s{}'.format(seed),
    load_path='/home/lihepeng/Documents/Github/matrpo/results/graphs/matrpo/s{}'.format(seed),
    info_keywords=tuple(['r{}'.format(i) for i in range(10)]+['overloading', 'overvoltage', 'undervoltage', 'total_loss']),
    mode=mode,
    **network_kwargs)

# training
total_timesteps = 500
for step in range(1, total_timesteps+1):
    actions, obs, rewards, returns, dones, values, advs, neglogpacs = agents.runner.run()
    agents.model.train(actions, obs, rewards, returns, dones, values, advs, neglogpacs)
    df_train = load_results('/home/lihepeng/Documents/Github/matrpo/results/rewards/matrpo/s{}'.format(seed),)
    if step % 10 == 1:
        plot(df_train, agents, 80)
        agents.model.save()





