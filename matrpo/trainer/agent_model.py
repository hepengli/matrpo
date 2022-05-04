from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time, os
from baselines.common.models import get_network_builder
from baselines.common.mpi_adam import MpiAdam
from baselines.common import colorize
from baselines.common.cg import cg
from matrpo.common.policies import PolicyWithValue
from matrpo.common.lbfgs import lbfgs

from contextlib import contextmanager
MPI = None

class AgentModel(tf.Module):
    def __init__(self, agent, network, nbatch, rho, max_kl, ent_coef, vf_stepsize, vf_iters,
                 cg_damping, cg_iters, lbfgs_iters, load_path, **network_kwargs):
        super(AgentModel, self).__init__(name='AgentModel')
        self.agent = agent
        self.comms = agent.comms
        self.nbs = agent.neighbors
        self.nbatch = nbatch
        self.rho = rho
        self.max_kl = max_kl
        self.ent_coef = ent_coef
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.lbfgs_iters = lbfgs_iters
        self.vf_stepsize = vf_stepsize
        self.vf_iters = vf_iters

        if MPI is not None:
            self.nworkers = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
        else:
            self.nworkers = 1
            self.rank = 0

        # Setup losses and stuff
        # ----------------------------------------
        ob_space = agent.observation_space
        ac_space = agent.action_space

        with tf.name_scope(agent.name):
            if isinstance(network, str):
                network = get_network_builder(network)(**network_kwargs)
            with tf.name_scope("pi"):
                pi_policy_network = network(ob_space.shape)
                pi_value_network = network(ob_space.shape)
                self.pi = pi = PolicyWithValue(ob_space, ac_space, pi_policy_network, pi_value_network)
            with tf.name_scope("oldpi"):
                old_pi_policy_network = network(ob_space.shape)
                old_pi_value_network = network(ob_space.shape)
                self.oldpi = oldpi = PolicyWithValue(ob_space, ac_space, old_pi_policy_network, old_pi_value_network)

        self.comm_matrix = agent.comm_matrix.copy()
        self.estimates = np.zeros([agent.nmates, agent.action_size, nbatch], dtype=np.float32)
        self.multipliers = np.ones([agent.nmates, agent.action_size, nbatch], dtype=np.float32)

        pi_var_list = pi_policy_network.trainable_variables + list(pi.pdtype.trainable_variables)
        old_pi_var_list = old_pi_policy_network.trainable_variables + list(oldpi.pdtype.trainable_variables)
        vf_var_list = pi_value_network.trainable_variables + pi.value_fc.trainable_variables
        old_vf_var_list = old_pi_value_network.trainable_variables + oldpi.value_fc.trainable_variables

        self.pi_var_list = pi_var_list
        self.old_pi_var_list = old_pi_var_list
        self.vf_var_list = vf_var_list
        self.old_vf_var_list = old_vf_var_list

        if load_path is not None:
            load_path = os.path.expanduser(load_path)
            self.ckpt = tf.train.Checkpoint(model=pi)
            load_path = os.path.join(load_path, 'agent{}'.format(self.agent.id))
            self.manager = tf.train.CheckpointManager(self.ckpt, load_path, max_to_keep=3)
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

        self.vfadam = MpiAdam(vf_var_list)
        self.vfadam.sync()

        self.get_flat = U.GetFlat(pi_var_list)
        self.get_old_flat = U.GetFlat(old_pi_var_list)
        self.set_from_flat = U.SetFromFlat(pi_var_list)
        self.loss_names = ["Lagrange", "surrgain", "sync", "meankl", "entloss", "entropy"]
        self.shapes = [var.get_shape().as_list() for var in pi_var_list]

    def reinitial_estimates(self):
        self.estimates = np.zeros([self.agent.nmates, self.agent.action_size, self.nbatch], dtype=np.float32)
        self.multipliers = np.ones([self.agent.nmates, self.agent.action_size, self.nbatch], dtype=np.float32)

    def assign_old_eq_new(self):
        for pi_var, old_pi_var in zip(self.pi_var_list, self.old_pi_var_list):
            old_pi_var.assign(pi_var)
        for vf_var, old_vf_var in zip(self.vf_var_list, self.old_vf_var_list):
            old_vf_var.assign(vf_var)

    def assign_new_eq_old(self):
        for old_pi_var, pi_var in zip(self.old_pi_var_list, self.pi_var_list):
            pi_var.assign(old_pi_var)

    def convert_to_tensor(self, args):
        return [tf.convert_to_tensor(arg, dtype=arg.dtype) for arg in args]

    @contextmanager
    def timed(self, msg, verbose=False):
        if self.rank == 0:
            if verbose: print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            if verbose: print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(self, x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= self.nworkers
        else:
            out = np.copy(x)

        return out

    @tf.function
    def compute_losses(self, ob, ac, atarg, comms, estimates, multipliers):
        old_policy_latent = self.oldpi.policy_network(ob)
        old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
        policy_latent = self.pi.policy_network(ob)
        pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
        kloldnew = old_pd.kl(pd)
        ent = pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = self.ent_coef * meanent
        ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
        surrgain = tf.reduce_mean(ratio * atarg)
        logratio = pd.logp_n(ac) - old_pd.logp_n(ac)
        syncerr = tf.multiply(comms, tf.tile(logratio[None,:], comms.shape)) - estimates
        syncloss = tf.reduce_mean(tf.reduce_sum(multipliers * syncerr, axis=[0,1]) + \
                                  0.5 * self.rho * tf.reduce_sum(tf.square(syncerr), axis=[0,1]))
        lagrangeloss = - surrgain - entbonus + syncloss
        mean_syncerr = tf.reduce_mean(tf.reduce_sum(tf.square(syncerr), axis=[0,1]), axis=-1)
        losses = [lagrangeloss, surrgain, mean_syncerr, meankl, entbonus]
        return losses

    #ob shape should be [batch_size, ob_dim], merged nenv
    #ret shape should be [batch_size]
    @tf.function
    def compute_vflossandgrad(self, ob, ret):
        with tf.GradientTape() as tape:
            pi_vf = self.pi.value(ob)
            vferr = tf.reduce_mean(tf.square(pi_vf - ret))
        return U.flatgrad(tape.gradient(vferr, self.vf_var_list), self.vf_var_list)

    @tf.function
    def compute_fvp(self, flat_tangents, ob, ac):
        tangents = self.reshape_from_flat(flat_tangents)
        with tf.autodiff.ForwardAccumulator(
            primals=self.pi_var_list,
            # The "vector" in Hessian-vector product.
            tangents=tangents) as acc:
            with tf.GradientTape() as tape:
                old_policy_latent = self.oldpi.policy_network(ob)
                old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
                policy_latent = self.pi.policy_network(ob)
                pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
                kloldnew = old_pd.kl(pd)
                meankl = tf.reduce_mean(kloldnew)
            backward = tape.gradient(meankl, self.pi_var_list)
        fvp = U.flatgrad(acc.jvp(backward), self.pi_var_list)

        return fvp

    @tf.function
    def compute_jjvp(self, flat_tangents, ob, ac, atarg):
        tangents = self.reshape_from_flat(flat_tangents)
        with tf.autodiff.ForwardAccumulator(
                primals=self.pi_var_list,
                tangents=tangents) as acc:
            old_policy_latent = self.oldpi.policy_network(ob)
            old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
            policy_latent = self.pi.policy_network(ob)
            pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
            logratio = pd.logp(ac) - old_pd.logp(ac)
        jvp = acc.jvp(logratio)

        with tf.GradientTape() as tape:
            old_policy_latent = self.oldpi.policy_network(ob)
            old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
            policy_latent = self.pi.policy_network(ob)
            pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
            logratio = pd.logp(ac) - old_pd.logp(ac)
            jvpr = tf.reduce_mean(jvp * logratio)
        jjvp = U.flatgrad(tape.jacobian(jvpr, self.pi_var_list), self.pi_var_list)

        return jjvp

    @tf.function
    def compute_vjp(self, ob, ac, atarg, comms, estimates, multipliers):
        atargs = tf.tile(tf.expand_dims(atarg, 0), [self.agent.action_size, 1])
        with tf.GradientTape() as tape:
            old_policy_latent = self.oldpi.policy_network(ob)
            old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
            policy_latent = self.pi.policy_network(ob)
            pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
            ent = - tf.exp(old_pd.logp(ac)) * (old_pd.logp(ac) + 1.)
            ent_bonus = self.ent_coef * tf.reduce_mean(ent)
            logratio = pd.logp_n(ac) - old_pd.logp_n(ac)
            v = atargs - tf.reduce_sum(comms * multipliers, axis=0) + \
                self.rho * tf.reduce_sum(comms * estimates, axis=0)
            vpr = tf.reduce_mean(tf.reduce_sum(v * logratio, axis=0)) + ent_bonus
        vjp = tape.jacobian(vpr, self.pi_var_list)

        return U.flatgrad(vjp, self.pi_var_list)

    @tf.function
    def compute_jvp(self, ob, ac, flat_tangents):
        tangents = self.reshape_from_flat(flat_tangents)
        with tf.autodiff.ForwardAccumulator(
                primals=self.pi_var_list,
                tangents=tangents) as acc:
            old_policy_latent = self.oldpi.policy_network(ob)
            old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
            policy_latent = self.pi.policy_network(ob)
            pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
            logratio = pd.logp_n(ac) - old_pd.logp_n(ac)

        return acc.jvp(logratio)

    @tf.function
    def compute_logratio(self, ob, ac):
        old_policy_latent = self.oldpi.policy_network(ob)
        old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
        policy_latent = self.pi.policy_network(ob)
        pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
        logratio = pd.logp_n(ac) - old_pd.logp_n(ac)

        return logratio

    @tf.function
    def compute_lossandgrad(self, ob, ac, atarg):
        with tf.GradientTape() as tape:
            old_policy_latent = self.oldpi.policy_network(ob)
            old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
            policy_latent = self.pi.policy_network(ob)
            pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
            kloldnew = old_pd.kl(pd)
            ent = pd.entropy()
            meankl = tf.reduce_mean(kloldnew)
            meanent = tf.reduce_mean(ent)
            entbonus = self.ent_coef * meanent
            ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
            surrgain = tf.reduce_mean(ratio * atarg)
            optimgain = surrgain + entbonus
            losses = [optimgain, meankl, entbonus, surrgain, meanent]
        gradients = tape.gradient(optimgain, self.pi_var_list)
        return losses + [U.flatgrad(gradients, self.pi_var_list)]

    @tf.function
    def trpo_compute_losses(self, ob, ac, atarg):
        old_policy_latent = self.oldpi.policy_network(ob)
        old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
        policy_latent = self.pi.policy_network(ob)
        pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
        kloldnew = old_pd.kl(pd)
        ent = pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = self.ent_coef * meanent
        ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
        surrgain = tf.reduce_mean(ratio * atarg)
        optimgain = surrgain + entbonus
        losses = [optimgain, meankl, entbonus, surrgain, meanent]
        return losses

    @tf.function
    def trpo_compute_fvp(self, flat_tangent, ob, ac, atarg):
        with tf.GradientTape() as outter_tape:
            with tf.GradientTape() as inner_tape:
                old_policy_latent = self.oldpi.policy_network(ob)
                old_pd, _ = self.oldpi.pdtype.pdfromlatent(old_policy_latent)
                policy_latent = self.pi.policy_network(ob)
                pd, _ = self.pi.pdtype.pdfromlatent(policy_latent)
                kloldnew = old_pd.kl(pd)
                meankl = tf.reduce_mean(kloldnew)
            klgrads = inner_tape.gradient(meankl, self.pi_var_list)
            start = 0
            tangents = []
            for shape in self.shapes:
                sz = U.intprod(shape)
                tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
                start += sz
            gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])
        hessians_products = outter_tape.gradient(gvp, self.pi_var_list)
        fvp = U.flatgrad(hessians_products, self.pi_var_list)
        return fvp

    def reshape_from_flat(self, flat_tangents):
        shapes = [var.get_shape().as_list() for var in self.pi_var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangents[start:start+sz], shape))
            start += sz
        
        return tangents

    def info_to_exchange(self, ob, ac, nb):
        # delta = self.get_flat()-self.get_old_flat()
        # logratio = self.compute_jvp(ob, ac, delta).numpy()
        logratio = self.compute_logratio(ob, ac).numpy()
        multiplier = self.multipliers[nb].copy()

        return logratio, multiplier

    def exchange(self, ob, ac, comm, nb_logratio, nb_multipliers, nb):
        # delta = self.get_flat()-self.get_old_flat()
        # logratio = self.compute_jvp(ob, ac, delta).numpy()
        logratio = self.compute_logratio(ob, ac).numpy()
        multiplier = self.multipliers[nb].copy()
        v = 0.5 * (multiplier + nb_multipliers) + \
            0.5 * self.rho * (comm * logratio + (-comm) * nb_logratio)
        self.estimates[nb] = (multiplier - v) / self.rho + comm * logratio
        self.multipliers[nb] = v.copy()

    def update(self, obs, actions, atarg, returns, vpredbefore):
        # Prepare data
        args = self.convert_to_tensor((obs, actions, atarg))
        fvpargs = self.convert_to_tensor((obs[::5], actions[::5]))
        synargs = self.convert_to_tensor((self.comms, self.estimates[self.nbs], self.multipliers[self.nbs]))

        self.assign_new_eq_old()

        def hvp(p):
            fvp = self.allmean(self.compute_fvp(p, *fvpargs).numpy())
            # jjvp = self.allmean(self.compute_jjvp(p, *fvpargs).numpy())
            return fvp + self.cg_damping * p

        with self.timed("computegrad"):
            g = self.allmean(self.compute_vjp(*args, *synargs).numpy())
        lossesbefore = self.allmean(np.array(self.compute_losses(*args, *synargs)))

        if np.allclose(g, 0):
            # logger.log("Got zero gradient. not updating")
            return False
        else:
            with self.timed("cg"):
                stepdir = cg(hvp, g, cg_iters=self.cg_iters)
            assert np.isfinite(stepdir).all()
            shs = 0.5*g.dot(stepdir)
            lm = np.sqrt(shs / self.max_kl)
            logger.log("---------------- {} ----------------".format(self.agent.name))
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            # expectedimprove = g.dot(fullstep)
            lagrangebefore, surrbefore, *_, entbefore = lossesbefore
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                losses = lagrange, surr, syncerr, kl, ent = self.allmean(np.array(self.compute_losses(*args, *synargs)))
                improve = lagrangebefore - lagrange
                surr_improve = surr - surrbefore
                logger.log("Surr_improve: %.5f Sync_error: %.5f"%(surr_improve, syncerr))
                # logger.log("Entropy before: %.5f Entropy: %.5f"%(entbefore, ent))
                # logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(losses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > self.max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("lagrange didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                self.set_from_flat(thbefore)

    def trpo_update(self, obs, actions, atarg, returns, vpredbefore):
        # Prepare data
        args = (obs, actions, atarg)
        # Sampling every 5
        fvpargs = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return self.allmean(self.trpo_compute_fvp(p, *fvpargs).numpy()) + self.cg_damping * p

        with self.timed("computegrad"):
            *lossbefore, g = self.compute_lossandgrad(*args)
        lossbefore = self.allmean(np.array(lossbefore))
        g = g.numpy()
        g = self.allmean(g)

        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with self.timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=self.cg_iters)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self.max_kl)
            logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                meanlosses = surr, kl, *_ = self.allmean(np.array(self.trpo_compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > self.max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                self.set_from_flat(thbefore)

    def vfupdate(self, obs, returns, values):
        with self.timed("vf"):
            for _ in range(self.vf_iters):
                for (mbob, mbret) in dataset.iterbatches((obs, returns),
                include_final_partial_batch=False, batch_size=64):
                    g = self.compute_vflossandgrad(mbob, mbret)
                    g = self.allmean(g.numpy())
                    self.vfadam.update(g, self.vf_stepsize)

        logger.log("ev_tdlam_before", explained_variance(values, returns))

