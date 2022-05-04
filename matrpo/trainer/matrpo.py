import numpy as np
import os, gym

from baselines.common import set_global_seeds
from matrpo.trainer.model import Model
from matrpo.trainer.runner import Runner
from matrpo.trainer.build_policy import Policy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MATRPO(object):
    """ MATRPO algorithm """
    def __init__(self, env, adj_matrix, nsteps, network, admm_iter, num_env, seed=None, test_env=None,
                 finite=True, load_path=None, logger_dir=None, force_dummy=False, mode='matrpo', 
                 gamma=0.995, lam=0.95, max_kl=0.001, ent_coef=0.0, vf_stepsize=3e-4, vf_iters=3, 
                 cg_damping=1e-2, cg_iters=10, lbfgs_iters=10, rho=1.0, reward_scale=1.0, 
                 ob_normalization=False, info_keywords=(), **network_kwargs):

        set_global_seeds(seed)
        np.set_printoptions(precision=5)
        nbatch = num_env * nsteps
        self.env = env
        self.test_env = test_env

        # create interactive policies for each agent
        self.policies = policies = Policy(env, adj_matrix, network, nbatch, mode, rho, max_kl, ent_coef, 
                vf_stepsize, vf_iters, cg_damping, cg_iters, lbfgs_iters, load_path, **network_kwargs)

        # model
        self.model = model = Model(env, policies, admm_iter, mode, ob_normalization)

        # runner
        self.runner = Runner(env, model, nsteps, gamma, lam, finite)

    def evaluate(self, n_episodes, save_replay=False):
        self.model.test = True
        for _ in range(n_episodes):
            obs_n = self.test_env.reset()
            while True:
                # query for action from each agent's policy
                act_n, _, _ = self.model.step(obs_n)
                # step environment
                obs_n, reward_n, done_n, info_n = self.test_env.step(act_n)
                # break
                if any(done_n):
                    print('done!')
                    break

        if save_replay:
            self.test_env.unwrapped.save_replay()

        # Finish test
        self.model.test = False
        self.test_env.unwrapped.close()

    def play(self):
        if self.test_env:
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