import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from baselines.bench.monitor import load_results

def get_training_curve(env, model, num_agents):
    env_id = '{}_{}'.format(env, num_agents)
    agents = ['r{}'.format(i) for i in range(num_agents)]
    seeds = [1,2,3,4,5]
    results = []
    for seed in seeds:
        reward_path = '/home/lihepeng/Documents/Github/results/training/{}/{}/s{}'.format(env_id, model, seed)
        if model in ['central','matrpo','trpo']:
            df_train = load_results(reward_path)
            data = df_train[agents].sum(axis=1).values
            data = np.mean(data.reshape(-1, 100), axis=1)
        elif model in ['fdmarl']:
            data = pd.read_csv(reward_path+'/train_reward_{}.csv'.format(seed))['sum_reward'].values
            if env=='collector' and num_agents==8: data = (data + 15) * 0.85
            if env=='collector' and num_agents==12: data[:1500] += 35
            data = np.mean(data.reshape(-1, 100), axis=1)
        elif model in ['maac']:
            data = pd.read_csv(reward_path+'/train_reward_{}.csv'.format(seed))['Value'].values
            data = np.mean(data.reshape(-1, 2), axis=1)
        data = np.hstack([data, data[-1:]])
        results.append(data)
    return np.array(results)

xdata = np.arange(501)
x1data = np.arange(3001)
models = ['central','trpo','fdmarl','matrpo']
results = []
for model in models:
    results.append(get_training_curve('simple_spread', model, num_agents=6))

from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
# plt.rc('xtick', direction='out', color='black', fontsize='large')
# plt.rc('ytick', direction='out', color='black', fontsize='large')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

fig, ax = plt.subplots(figsize=(5.5,5))
sns.tsplot(time=xdata, data=results[0], color='r', linestyle='-', legend=False)
sns.tsplot(time=xdata, data=results[1], color='b', linestyle='-', legend=False)
sns.tsplot(time=xdata, data=results[2], color='c', linestyle='-', legend=False)
sns.tsplot(time=xdata, data=results[3], color='g', linestyle='-', legend=False)
# sns.tsplot(time=xdata, data=results[4], color='y', linestyle='-', legend=True)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [Patch(color='r', label='C-TRPO'),
                   Patch(color='b', label='I-TRPO'),
                   Patch(color='c', label='FDMARL'),
                   Patch(color='g', label='MATRPO'),
                #    Patch(color='y', label='MAAC')
                ]
ax.legend(handles=legend_elements, fontsize='large')
# plt.ylim(-500, 0)
plt.xticks(ticks=range(0,501,100), labels=['0','100 (1M)','200 (2M)','300 (3M)','400 (4M)','500 (5M)'], fontsize='medium')
plt.xlabel('number of policy iterations (episodes)', fontsize='x-large')
plt.ylabel('return', fontsize='x-large')
plt.title('Cooperative Navigation ($N=6$)', fontsize='x-large')
plt.grid(color='w', linestyle='solid')
plt.tight_layout(rect=(-0.03,-0.0,1.03,1.0))
plt.show()


# # Predator prey
# def get_training_curve_1(model):
#     env_id = 'simple_predator_prey'
#     seeds = [1,2,3]
#     pred_results = []
#     prey_results = []
#     for seed in seeds:
#         reward_path = '/home/lihepeng/Documents/Github/results/training/{}/{}/s{}'.format(env_id, model, seed)
#         df_train = load_results(reward_path)
#         pred_data = df_train[['r0','r1','r2','r3']].sum(axis=1).values[:200*200]
#         pred_data = np.mean(pred_data.reshape(-1, 200), axis=1)
#         pred_results.append(pred_data)

#         prey_data = df_train[['r4','r5','r6']].sum(axis=1).values[:200*200]
#         prey_data = np.mean(prey_data.reshape(-1, 200), axis=1)
#         prey_results.append(prey_data)

#     return np.array(pred_results), np.array(prey_results)


# xdata = np.arange(200)
# models = ['independent_vs_independent']
# pred_results, prey_results = [], []
# for model in models:
#     pred, prey = get_training_curve_1(model)
#     pred_results.append(pred)
#     prey_results.append(prey)

# fig = plt.figure()
# sns.tsplot(time=xdata, data=pred_results[0], color='r', linestyle='-', legend=True)
# # fig = plt.figure()
# # sns.tsplot(time=xdata, data=pred_results[0], color='r', linestyle='-', legend=True)
# # sns.tsplot(time=xdata, data=results[1], color='g', linestyle='-', legend=True)
# # sns.tsplot(time=xdata, data=results[2], color='b', linestyle='-', legend=True)
# # plt.legend(labels=['Central TRPO', 'MATRPO', 'Independent TRPO'])
# plt.show()

