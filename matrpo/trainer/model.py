import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, env, policies, admm_iter, mode, adj_matrix, ob_normalization):
        self.env = env
        self.policies = policies
        self.admm_iter = admm_iter 
        self.mode = mode
        self.adj_matrix = adj_matrix
        self.leader = 0 if mode == 'central' else None
        self.test = False
        self.ob_normalization = ob_normalization

    def step(self, obs):
        if not self.test:
            return self.train_step(obs)
        else:
            return self.test_step(obs)

    def train_step(self, obs):
        actions, values, neglogps = [], [], []
        for i, policy in enumerate(self.policies):
            obs_i = np.stack(obs[:,i], axis=0)
            action, value, _, neglogp = self.policies[i].pi.step(obs_i)
            # extract local action
            indices = list(policy.agent.action_index)
            if self.mode=='matrpo':
                action_i = tf.gather(action, indices, axis=1).numpy()
                actions.extend(np.split(action_i, len(indices), axis=1))
            elif self.mode=='trpo':
                actions.append(action.numpy().squeeze().transpose())
            elif self.mode == 'central':
                if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                if i == self.leader: actions += list(action.numpy().transpose())
            values.append(value.numpy())
            neglogps.append(neglogp.numpy())

        actions = np.stack(actions, axis=1).squeeze()
        values = np.stack(values, axis=1)
        neglogps = np.stack(neglogps, axis=1)

        # # Homomorphic Encryption
        # sum_values = np.zeros_like(values)
        # for i in range(values.shape[1]):
        #     sum_values[:,i] = values.sum(axis=1)

        return actions, values, neglogps

    def test_step(self, obs):
        actions, values, neglogps = [], [], []
        for i, policy in enumerate(self.policies):
            obs_i = np.expand_dims(obs[i], axis=0)
            action, value, _, neglogp = policy.pi.step(obs_i)
            if self.mode=='matrpo':
                if len(action.shape) < 2:
                    action = tf.expand_dims(action, axis=1)
                actions.append(action[0,i].numpy())
            elif self.mode=='trpo':
                actions.append(action.numpy().squeeze())
            elif self.mode=='central':
                if len(action.shape) < 2: action = tf.expand_dims(action, axis=1)
                if i == self.leader: actions += list(action[0].numpy())
            values.append(value)
            neglogps.append(neglogp)

        return actions, values, neglogps

    def value(self, obs):
        return np.vstack([policy.pi.value(tf.stack(obs[:,i], axis=0)) 
                            for i, policy in enumerate(self.policies)]).transpose()

    def share_actions(self, actions):
        shared_actions_n = []
        for i in range(len(self.env.action_space)):
            if self.mode == 'matrpo':
                shared_actions_n.append(actions)
            elif self.mode == 'trpo':
                shared_actions_n.append(actions[:,i:i+1])
            elif self.mode == 'central':
                shared_actions_n.append(actions)

        return shared_actions_n

    def save(self):
        for pi in self.policies:
            save_path = pi.manager.save()
            print("Save checkpoint to {}".format(save_path))

    def train(self, actions, obs, rewards, returns, dones, values, advs, neglogpacs):
        eps = 1e-8
        edges = self.adj_matrix[np.unique(np.nonzero(self.adj_matrix)[0])]
        # Policy Update
        if self.mode == 'matrpo':
            for i in range(len(self.env.action_space)):
                if self.ob_normalization:
                    self.policies[i].pi.ob_rms.update(obs[i])
                    self.policies[i].oldpi.ob_rms.update(obs[i])
                self.policies[i].reinitial_estimates()
                self.policies[i].assign_old_eq_new()
                self.policies[i].vfupdate(obs[i], returns[i], values[i])
            # prepare data
            norm_advs = [(adv-np.mean(advs))/(np.std(advs)+eps) for adv in advs]
            argvs = tuple(zip(obs, actions, norm_advs, returns, values))
            # consensus using admm
            for itr in range(self.admm_iter):
                # edge = edges[np.random.choice(range(len(edges)))]
                edge = edges[itr % len(edges)]
                q = np.where(edge != 0)[0]
                k, j = q[0], q[-1]
                # Update Agent k and j
                self.policies[k].update(*argvs[k])
                self.policies[j].update(*argvs[j])
                ratio_k, multipliers_k = self.policies[k].info_to_exchange(obs[k], actions[k], j)
                ratio_j, multipliers_j = self.policies[j].info_to_exchange(obs[j], actions[j], k)
                self.policies[k].exchange(obs[k], actions[k], edge[k], ratio_j, multipliers_j, j)
                self.policies[j].exchange(obs[j], actions[j], edge[j], ratio_k, multipliers_k, k)
        elif self.mode == 'central':
            if self.ob_normalization:
                self.policies[self.leader].pi.ob_rms.update(obs[self.leader])
                self.policies[self.leader].oldpi.ob_rms.update(obs[self.leader])
            norm_advs = (advs[self.leader] - np.mean(advs[self.leader])) / (np.std(advs[self.leader])+eps)
            argvs = (obs[self.leader], actions[self.leader], norm_advs, returns[self.leader], values[self.leader])
            self.policies[self.leader].assign_old_eq_new()
            self.policies[self.leader].vfupdate(obs[self.leader], returns[self.leader], values[self.leader])
            self.policies[self.leader].trpo_update(*argvs)
        else:
            norm_advs = advs.copy()
            for i in range(len(self.env.action_space)):
                if self.ob_normalization:
                    self.policies[i].pi.ob_rms.update(obs[i])
                    self.policies[i].oldpi.ob_rms.update(obs[i])
                norm_advs[i] = (advs[i]-np.mean(advs[i]))/(np.std(advs[i])+eps)
                self.policies[i].assign_old_eq_new()
                self.policies[i].vfupdate(obs[i], returns[i], values[i])
                self.policies[i].trpo_update(obs[i], actions[i], norm_advs[i], returns[i], values[i])
