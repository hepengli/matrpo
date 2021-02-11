import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(df_train, agents, episodes):
    rewards = df_train['r'].values
    rewards = np.mean(rewards.reshape(-1, episodes), axis=1)
    plt.close()
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(rewards)
    ax.set_title('Rewards')
    plt.tight_layout(rect=(0,0,1,1))
    plt.show(block=False)
    plt.pause(0.5)
