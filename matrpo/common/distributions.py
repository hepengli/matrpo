import tensorflow as tf
import numpy as np
from gym import spaces
from baselines.a2c.utils import fc

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def logp(self, x):
        return - self.neglogp(x)
    def get_shape(self):
        return self.flatparam().shape
    @property
    def shape(self):
        return self.get_shape()
    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])

class PdType(tf.Module):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def pdfromlatent(self, latent_vector):
        raise NotImplementedError
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)

class CategoricalPdType(PdType):
    def __init__(self, latent_shape, ncat, init_scale=1.0, init_bias=0.0):
        self.ncat = ncat
        self.matching_fc = _matching_fc(latent_shape, 'pi', self.ncat, init_scale=init_scale, init_bias=init_bias)

    def pdclass(self):
        return CategoricalPd
    def pdfromlatent(self, latent_vector):
        pdparam = self.matching_fc(latent_vector)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return tf.int32

class MultiCategoricalPdType(PdType):
    def __init__(self, latent_shape, nvec, init_scale=1.0, init_bias=0.0):
        self.ncats = nvec.astype('int32')
        self.matching_fc = _matching_fc(latent_shape, 'pi', self.ncats.sum(), init_scale=init_scale, init_bias=init_bias)
        assert (self.ncats > 0).all()

    def pdclass(self):
        return MultiCategoricalPd
    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.ncats, flat)

    def pdfromlatent(self, latent_vector):
        pdparam = self.matching_fc(latent_vector)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [len(self.ncats)]
    def sample_dtype(self):
        return tf.int32

class DiagGaussianPdType(PdType):
    def __init__(self, latent_shape, size, init_scale=1.0, init_bias=0.0):
        self.size = size
        self.matching_fc = _matching_fc(latent_shape, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        self.logstd = tf.Variable(np.zeros((1, self.size)), name='pi/logstd', dtype=tf.float32)

    def pdclass(self):
        return DiagGaussianPd

    def pdfromlatent(self, latent_vector):
        mean = self.matching_fc(latent_vector)
        pdparam = tf.concat([mean, mean * 0.0 + self.logstd], axis=1)
        return self.pdfromflat(pdparam), mean

    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32


class MixedPdType(PdType):
    def __init__(self, latent_shape, action_spaces, init_scale=1.0, init_bias=0.0):
        shape, size, matching_fc = [], [], []
        for i, ac_space in enumerate(action_spaces):
            if isinstance(ac_space, spaces.Box):
                shape.append(ac_space.shape[0])
                size.append(2 * ac_space.shape[0])
                mean = _matching_fc(latent_shape, 'pi/mean_{}'.format(i), ac_space.shape[0], init_scale=init_scale, init_bias=init_bias)
                logstd = tf.Variable(np.zeros((1, ac_space.shape[0])), name='pi/logstd_{}'.format(i), dtype=tf.float32)
                matching_fc.append([mean, logstd])
            elif isinstance(ac_space, spaces.Discrete):
                shape.append(1)
                size.append(ac_space.n)
                matching_fc.append(_matching_fc(
                    latent_shape, 'pi/logits_{}'.format(i), ac_space.n, init_scale=init_scale, init_bias=init_bias))
            elif isinstance(ac_space, spaces.MultiDiscrete):
                shape.append(len(ac_space.nvec))
                size.append(sum(ac_space.nvec))
                matching_fc.append(_matching_fc(
                    latent_shape, 'pi/multilogits_{}'.format(i), sum(ac_space.nvec), init_scale=init_scale, init_bias=init_bias))
            else:
                raise NotImplementedError
        self.shape = shape
        self.size = size
        self.matching_fc = matching_fc
        self.action_spaces = action_spaces
    def pdclass(self):
        return MixedPd
    def pdfromflat(self, flat):
        return MixedPd(flat)
    def param_shape(self):
        return [sum(self.size)]
    def sample_shape(self):
        return [sum(self.shape)]
    def sample_dtype(self):
        return tf.float32
    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = {}
        pdparam["flat"] = []
        # print(self.matching_fc.trainable_variables)
        for mfc in self.matching_fc:
            if isinstance(mfc, list):
                mean, logstd = mfc[0](latent_vector), mfc[1]
                pdparam["flat"].append(tf.concat([mean, mean * 0.0 + logstd], axis=1))
            else:
                pdparam["flat"].append(mfc(latent_vector))
        pdparam["ac_spaces"] = self.action_spaces
        pdparam["nvec"] = self.shape
        return self.pdfromflat(pdparam), pdparam


class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return tf.nn.softmax(self.logits)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            # already encoded
            print('logits is {}'.format(self.logits))
            assert x.shape.as_list() == self.logits.shape.as_list()

        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=x)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)
    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
    def sample(self):
        u = tf.random.uniform(tf.shape(self.logits), dtype=self.logits.dtype, seed=0)
        return tf.argmax(self.logits - tf.math.log(-tf.math.log(u)), axis=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class MultiCategoricalPd(Pd):
    def __init__(self, nvec, flat):
        self.flat = flat
        self.categoricals = list(map(CategoricalPd,
            tf.split(flat, np.array(nvec, dtype=np.int32), axis=-1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)
    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])
    def logp_n(self, x):
        return tf.stack([- p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))], axis=0)
    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])
    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])
    def sample(self):
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)
    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.mean
    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], dtype=tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    def sample(self):
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class MixedPd(Pd):
    def __init__(self, flat):
        self.nvec = flat["nvec"]
        self.pds, self.dtypes = [], []
        for ac_space, param in zip(flat["ac_spaces"], flat["flat"]):
            if isinstance(ac_space, spaces.Box):
                self.pds.append(DiagGaussianPd(param))
                self.dtypes.append(tf.dtypes.as_dtype(ac_space.dtype))
            elif isinstance(ac_space, spaces.Discrete):
                self.pds.append(CategoricalPd(param))
                self.dtypes.append(tf.int32)
            elif isinstance(ac_space, spaces.MultiDiscrete):
                self.pds.append(MultiCategoricalPd(ac_space.nvec, param))
                self.dtypes.append(tf.int32)
    def flatparam(self):
        raise NotImplementedError
        # return self.flat
    def mode(self):
        return tf.concat([pd.mode() for pd in self.pds], axis=1)
    def logp_n(self, x):
        return tf.stack([- p.neglogp(tf.cast(px, tx)) for p, tx, px in zip(self.pds, self.dtypes, tf.split(x, np.array(self.nvec, dtype=np.int32), axis=-1))], axis=0)
    def neglogp(self, x):
        return tf.add_n([pd.neglogp(tf.cast(px, tx)) for pd, tx, px in zip(self.pds, self.dtypes, tf.split(x, np.array(self.nvec, dtype=np.int32), axis=-1))])
    def kl(self, other):
        return tf.add_n([pd.kl(otherpd) for pd, otherpd in zip(self.pds, other.pds)])
    def entropy(self):
        return tf.add_n([pd.entropy() for pd in self.pds])
    def sample(self):
        sample_acs = []
        for pd in self.pds:
            ac = tf.cast(pd.sample(), dtype=tf.float32)
            if len(ac.shape) < 2:
                ac = tf.expand_dims(ac, axis=1)
            sample_acs.append(ac)
        return tf.concat(sample_acs, axis=1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def make_pdtype(latent_shape, ac_space, init_scale=1.0):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(latent_shape, ac_space.shape[0], init_scale)
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(latent_shape, ac_space.n, init_scale)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(latent_shape, ac_space.nvec, init_scale)
    elif isinstance(ac_space, spaces.Tuple):
        return MixedPdType(latent_shape, ac_space.spaces, init_scale)
    else:
        raise ValueError('No implementation for {}'.format(ac_space))

def _matching_fc(tensor_shape, name, size, init_scale, init_bias):
    if tensor_shape[-1] == size:
        return lambda x: x
    else:
        return fc(tensor_shape, name, size, init_scale=init_scale, init_bias=init_bias)
