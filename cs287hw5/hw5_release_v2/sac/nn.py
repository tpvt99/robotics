import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions
from tensorflow.python import keras


class QFunction(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_sizes, **kwargs):
        super(QFunction, self).__init__(**kwargs)
        self._hidden_layer_sizes = hidden_layer_sizes

        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def build(self, input_shape):
        self.q_function = tf.keras.Sequential()
        obs_shape, act_shape = input_shape

        if isinstance(obs_shape, int):
            self.q_function.add(tf.keras.Input(shape = (obs_shape + act_shape, )))
        elif isinstance(obs_shape, tuple):
            self.q_function.add(tf.keras.Input(shape = (obs_shape[0] + act_shape[0], )))

        for hidden_units in self._hidden_layer_sizes:
            self.q_function.add(tf.keras.layers.Dense(units=hidden_units, activation='relu'))

        self.q_function.add(tf.keras.layers.Dense(units=1, activation=None))
        self.built = True

    def call(self, inputs, **kwargs):
        obs_input, act_input = inputs
        out = self.concat([obs_input, act_input])
        out = self.q_function(out)
        return out


class ValueFunction(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_sizes, **kwargs):
        super(ValueFunction, self).__init__(**kwargs)
        self._hidden_layer_sizes = hidden_layer_sizes

    def build(self, input_shape):
        self.v_function = tf.keras.Sequential()

        if isinstance(input_shape, int):
            self.v_function.add(tf.keras.layers.Input(shape=(input_shape,)))
        elif isinstance(input_shape, tuple):
            self.v_function.add(tf.keras.layers.Input(shape=input_shape))

        for hidden_units in self._hidden_layer_sizes:
            self.v_function.add(tf.keras.layers.Dense(units=hidden_units, activation='relu'))

        self.v_function.add(tf.keras.layers.Dense(units=1, activation=None))
        self.built = True

    def call(self, inputs, **kwargs):
        obs = inputs
        out = self.v_function(obs)
        return out

class GaussianPolicy(tf.keras.layers.Layer):
    def __init__(self, action_dim, hidden_layer_sizes, reparameterize, **kwargs):
        super(GaussianPolicy, self).__init__(**kwargs)
        self._action_dim = action_dim
        self._f = None
        self._hidden_layer_sizes = hidden_layer_sizes
        self._reparameterize = reparameterize

    def build(self, input_shape):
        self.mean_and_log_std = tf.keras.Sequential()
        if isinstance(input_shape, int):
            self.mean_and_log_std.add(tf.keras.layers.Input(shape = (input_shape, )))
        elif isinstance(input_shape, tuple):
            self.mean_and_log_std.add(tf.keras.layers.Input(shape=input_shape))

        for hidden_units in self._hidden_layer_sizes:
            self.mean_and_log_std.add(tf.keras.layers.Dense(units=hidden_units, activation='relu'))

        self.mean_and_log_std.add(tf.keras.layers.Dense(units = self._action_dim * 2, activation=None))
        self.lambda_func = tf.keras.layers.Lambda(self.create_distribution_layer)
        self.built = True

    def create_distribution_layer(self, mean_and_log_std):
        mean, log_std = tf.split(
            mean_and_log_std, num_or_size_splits=2, axis=1)
        log_std = tf.clip_by_value(log_std, -20., 2.)

        distribution = distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(log_std))

        raw_actions = distribution.sample()
        if not self._reparameterize:
            """ YOUR CODE HERE FOR PROBLEM 3C --- provided """
            # hint: stop gradient for the raw actions
            raw_actions = tf.stop_gradient(raw_actions)
            pass
            """ YOUR CODE ENDS """
        log_probs = distribution.log_prob(raw_actions)
        log_probs -= self._squash_correction(raw_actions)

        actions = None
        """YOUR CODE HERE FOR PROBLEM 3D --- provided """
        # hint: clip the action with tanh
        actions = tf.tanh(raw_actions)
        """ YOUR CODE ENDS """
        return [actions, log_probs]

    def call(self, inputs, **kwargs):
        obs = inputs
        mean_and_log_std = self.mean_and_log_std(obs)

        samples, log_prob = self.lambda_func(mean_and_log_std)

        return samples, log_prob

    def _squash_correction(self, raw_actions):
        """
        :param raw_actions:
        :return:
        """
        """ YOUR CODE HERE FOR PROBLEM 3E --- provided"""
        result = tf.reduce_sum(
            2.0 * raw_actions + np.log(4.0) - 2.0 * tf.nn.softplus(2.0 * raw_actions),
            axis=1
        )
        """ YOUR CODE ENDS """
        return result

    def eval(self, observation):
        assert self.built and observation.ndim == 1
        action, _ = self.call(observation[None])
        return action.numpy().flatten()
