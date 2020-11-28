from hw5_tf2.dynamics.layers import MLP
import tensorflow as tf
import numpy as np
from hw5_tf2.utils.serializable import Serializable
from hw5_tf2.logger import logger
from hw5_tf2.dynamics.mlp_dynamics import MLPDynamicsModel
import time
from collections import OrderedDict
from hw5_tf2.dynamics.utils import normalize, denormalize, train_test_split


class MLPDynamicsEnsemble(MLPDynamicsModel):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 num_models=5,
                 hidden_sizes=(512, 512),
                 hidden_nonlinearity='swish',
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 weight_normalization=False,  # Doesn't work
                 normalize_input=True,
                 optimizer=tf.keras.optimizers.Adam,
                 valid_split_ratio=0.2,  # 0.1
                 rolling_average_persitency=0.99,
                 buffer_size=50000,
                 loss_str='MSE',
                 ):

        Serializable.quick_init(self, locals())

        max_logvar = 1
        min_logvar = 0.1

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.buffer_size_train = int(buffer_size * (1 - valid_split_ratio))
        self.buffer_size_test = int(buffer_size * valid_split_ratio)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_models = num_models
        self.hidden_sizes = hidden_sizes
        self.name = name
        self._dataset_train = None
        self._dataset_test = None
        self.loss_str = loss_str

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]
        self.timesteps_counter = 0
        self.used_timesteps_counter = 0

        self.hidden_nonlinearity = hidden_nonlinearity = self._activations[hidden_nonlinearity]
        self.output_nonlinearity = output_nonlinearity = self._activations[output_nonlinearity]

        self.optimizer = optimizer(learning_rate=self.learning_rate)

        """ computation graph for training and simple inference """
        with tf.name_scope(name) as scope:

            # create MLP
            mlps = []
            self.delta_preds = []
            self.obs_next_pred = []
            for i in range(num_models):
                with tf.name_scope(scope + 'model_{}'.format(i)) as inner_scope:
                    mlp = MLP(name = inner_scope,
                              output_dim=obs_space_dims,
                              hidden_sizes=hidden_sizes,
                              hidden_nonlinearity=hidden_nonlinearity,
                              output_nonlinearity=output_nonlinearity,
                              #input_var=obs_ph[i],
                              input_dim=obs_space_dims+action_space_dims,
                              )
                    mlps.append(mlp)

                self.delta_preds.append(mlp.output_var)

            # tensor_utils
            self.f_delta_pred = self.forward

        """ computation graph for inference where each of the models receives a different batch"""

        self.optimizer_lists = []
        for i in range(num_models):
            self.optimizer_lists.append(optimizer(learning_rate=learning_rate))

        self.f_delta_pred_model_batches = self.forward_with_different_batches

        self._networks = mlps
        # LayersPowered.__init__(self, [mlp.output_layer for mlp in mlps])

    """YOUR CODE HERE FOR PROBLEM 2C"""
    # hint: you should find the single model and look for what functions you need to implement.

    def forward_with_different_batches(self, observations, actions):
        obs_model_batches = tf.split(observations, self.num_models, axis=0)
        act_model_batches = tf.split(actions, self.num_models, axis=0)
        delta_predictions = []

        for i in range(self.num_models):
            nn_inputs = tf.concat([obs_model_batches[i], act_model_batches[i]], axis=1)
            delta_pred = self.delta_preds[i](nn_inputs)
            delta_predictions.append(delta_pred)

        delta_predictions = tf.concat(delta_predictions, axis=0)
        return delta_predictions

    def loss_with_different_batches(self, deltas_predictions, deltas):
        delta_model_batches = tf.split(deltas, self.num_models, axis=0)
        delta_pred_batches = tf.split(deltas_predictions, self.num_models, axis=0)
        losses = []

        for i in range(self.num_models):
            if self.loss_str == 'L2':
                loss = tf.reduce_mean(tf.linalg.norm(delta_model_batches[i] - delta_pred_batches[i], axis=1))
            elif self.loss_str == 'MSE':
                loss = tf.reduce_mean((delta_model_batches[i] - delta_pred_batches[i]) ** 2)

            losses.append(loss)

        return losses

    def fit_with_different_batches(self, observations, actions, deltas):
        with tf.GradientTape(persistent=True) as tape:
            delta_predictions = self.forward_with_different_batches(observations, actions)
            losses = self.loss_with_different_batches(delta_predictions, deltas)
        for i in range(self.num_models):
            gradient = tape.gradient(losses[i], self.delta_preds[i])
            self.optimizer_lists[i].apply_gradients(zip(gradient, self.delta_preds[i].trainable_variables))


    def forward(self, observations, actions):
        nn_inputs = tf.concat([observations, actions], axis=1)
        obs_ph = tf.split(nn_inputs, self.num_models, axis=0)

        delta_predictions = []
        for i in range(self.num_models):
            delta_predictions.append(self.delta_preds[i](obs_ph[i]))

        delta_predictions = tf.stack(delta_predictions, axis=2) # shape: (batch_size, ndim_obs, n_models)

        return delta_predictions



    def loss(self, delta_preds, delta_targets):
        if self.loss_str == 'L2':
            self.loss = tf.reduce_mean(tf.linalg.norm(delta_targets[:, :, None] - delta_preds, axis=1))
        elif self.loss_str == 'MSE':
            self.loss = tf.reduce_mean((delta_targets[:, :, None] - delta_preds) ** 2)
        else:
            raise NotImplementedError





