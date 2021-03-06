# from hw5.core import MLP
from hw5_tf2.dynamics.layers import MLP
from hw5_tf2.dynamics.utils import normalize, denormalize, train_test_split
import tensorflow as tf
import numpy as np
from hw5_tf2.utils.serializable import Serializable
from hw5_tf2.logger import logger
from collections import OrderedDict


class MLPDynamicsModel(Serializable):
    """
    Class for MLP continous dynamics model
    """
    _activations = {
        None: tf.identity,
        "relu": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x)
    }

    def __init__(self,
                 name,
                 env,
                 hidden_sizes=(500, 500),
                 hidden_nonlinearity="tanh",
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 weight_normalization=False,
                 normalize_input=True,
                 optimizer=tf.keras.optimizers.Adam,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 buffer_size=100000,
                 ):

        Serializable.quick_init(self, locals())

        self.normalization = None
        self.normalize_input = normalize_input
        self.use_reward_model = False
        self.buffer_size = buffer_size
        self.name = name
        self.hidden_sizes = hidden_sizes

        self._dataset_train = None
        self._dataset_test = None
        self.next_batch = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency
        self.hidden_nonlinearity = hidden_nonlinearity = self._activations[hidden_nonlinearity]
        self.output_nonlinearity = output_nonlinearity =self._activations[output_nonlinearity]

        with tf.name_scope(name) as scope:
            self.batch_size = batch_size
            self.learning_rate = learning_rate

            # determine dimensionality of state and action space
            self.obs_space_dims = env.observation_space.shape[0]
            self.action_space_dims = env.action_space.shape[0]

            # placeholders
            #self.obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_space_dims))
            #self.act_ph = tf.placeholder(tf.float32, shape=(None, self.action_space_dims))
            #self.delta_ph = tf.placeholder(tf.float32, shape=(None, self.obs_space_dims))

            self._create_stats_vars()

            # concatenate action and observation --> NN input
            #self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)

            # create MLP
            mlp = MLP(name=scope,
                      output_dim=self.obs_space_dims,
                      hidden_sizes=hidden_sizes,
                      hidden_nonlinearity=hidden_nonlinearity,
                      output_nonlinearity=output_nonlinearity,
                      input_dim=self.obs_space_dims + self.action_space_dims,
                      weight_normalization=weight_normalization)

            self.delta_pred = mlp.output_var

            # define loss and train_op
            """ YOUR CODE HERE FOR PROBLEM 2A.1, YOUR CODE HERE FOR PROBLEM 2B """
            # hint: you need to write the loss between delta_ph (your target) and delta_pred using a l2 loss.
            self.loss = self.loss_obj
            """ YOUR CODE ENDS """
            self.optimizer = optimizer(self.learning_rate)

            # tensor_utils
            self.f_delta_pred = self.delta_pred

        self._networks = [mlp]
        # LayersPowered.__init__(self, [mlp.output_layer])

    def forward(self, observations, actions, training = False):
        x = tf.concat([observations, actions], axis=1)
        x = self.delta_pred(x, training)
        return x

    def loss_obj(self, delta_pred, delta_target):
        #return tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(delta_pred - delta_target), axis=-1)))

        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(delta_pred - delta_target), axis=-1)))


    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True,
            verbose=False, valid_split_ratio=None,
            rolling_average_persitency=None, log_tabular=False, early_stopping=True, prefix='Model-'):
        """
        Fits the NN dynamics model
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after taking action - numpy array of shape (n_samples, ndim_obs)
        :param epochs: number of training epochs
        :param compute_normalization: boolean indicating whether normalization shall be (re-)computed given the data
        :param valid_split_ratio: relative size of validation split (float between 0.0 and 1.0)
        :param verbose: logging verbosity
        """
        assert obs.ndim == 2 and obs.shape[1]==self.obs_space_dims
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0


        # split into valid and test set
        """ YOUR CODE HERE FOR PROBLEM 2A.2 """
        # hint: here you need to compute delta which based on obs and obs_next
        delta = obs_next - obs
        """ YOUR CODE ENDS """
        obs_train, act_train, delta_train, obs_test, act_test, delta_test = train_test_split(obs, act, delta,
                                                                                             test_split_ratio=valid_split_ratio)

        if self._dataset_test is None:
            self._dataset_test = dict(obs=obs_test, act=act_test, delta=delta_test)
            self._dataset_train = dict(obs=obs_train, act=act_train, delta=delta_train)
        else:
            n_test_new_samples = len(obs_test)
            n_max_test = self.buffer_size - n_test_new_samples
            n_train_new_samples = len(obs_train)
            n_max_train = self.buffer_size - n_train_new_samples
            self._dataset_test['obs'] = np.concatenate([self._dataset_test['obs'][-n_max_test:], obs_test])
            self._dataset_test['act'] = np.concatenate([self._dataset_test['act'][-n_max_test:], act_test])
            self._dataset_test['delta'] = np.concatenate([self._dataset_test['delta'][-n_max_test:], delta_test])

            self._dataset_train['obs'] = np.concatenate([self._dataset_train['obs'][-n_max_train:], obs_train])
            self._dataset_train['act'] = np.concatenate([self._dataset_train['act'][-n_max_train:], act_train])
            self._dataset_train['delta'] = np.concatenate([self._dataset_train['delta'][-n_max_train:], delta_train])

        # create data queue


        valid_loss_rolling_average = None

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(self._dataset_train['obs'], self._dataset_train['act'],
                                       self._dataset_train['delta'])

        if self.normalize_input:
            # normalize data
            """  YOUR CODE HERE FOR PROBLEM 2A.3 --- PROVIDED"""
            # hint: no code here, you need to implement the following function
            obs_train, act_train, delta_train = self._normalize_data(self._dataset_train['obs'],
                                                                     self._dataset_train['act'],
                                                                     self._dataset_train['delta'])
            """ YOUR CODE ENDS """
            assert obs_train.ndim == act_train.ndim == delta_train.ndim == 2
        else:
            obs_train = self._dataset_train['obs']
            act_train = self._dataset_train['act']
            delta_train = self._dataset_train['delta']



        dataset =  self._data_input_fn(obs_train,act_train,
                                                delta_train,
                                                batch_size=self.batch_size,
                                                buffer_size=self.buffer_size)

        # Training loop
        for epoch in range(epochs):

            # initialize data queue
            # sess.run(self.iterator.initializer,
            #          feed_dict={self.obs_dataset_ph: obs_train,
            #                     self.act_dataset_ph: act_train,
            #                     self.delta_dataset_ph: delta_train})

            batch_losses = []
            iterator = iter(dataset)
            while True:
                try:
                    with tf.GradientTape() as tape:
                        obs_batch, act_batch, delta_batch = iterator.get_next()

                        # run train op
                        delta_prediction = self.forward(obs_batch, act_batch, training=True)
                        batch_loss = self.loss(delta_prediction, delta_batch)
                        batch_losses.append(batch_loss)
                    gradients = tape.gradient(batch_loss, self.delta_pred.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.delta_pred.trainable_variables))

                except tf.errors.OutOfRangeError:
                    # compute validation loss
                    if self.normalize_input:
                        # normalize data
                        obs_test, act_test, delta_test = self._normalize_data(self._dataset_test['obs'],
                                                                              self._dataset_test['act'],
                                                                              self._dataset_test['delta'])
                        assert obs_test.ndim == act_test.ndim == delta_test.ndim == 2
                    else:
                        obs_test = self._dataset_test['obs']
                        act_test = self._dataset_test['act']
                        delta_test = self._dataset_test['delta']

                    valid_loss = self.loss(self.forward(obs_test, act_test), delta_test)

                    # valid_loss = sess.run(self.loss, feed_dict={self.obs_ph: obs_test,
                    #                                             self.act_ph: act_test,
                    #                                             self.delta_ph: delta_test})

                    if valid_loss_rolling_average is None:
                        valid_loss_rolling_average = 1.5 * valid_loss # set initial rolling to a higher value avoid too early stopping
                        valid_loss_rolling_average_prev = 2.0 * valid_loss

                    valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average + (1.0-rolling_average_persitency)*valid_loss

                    if verbose:
                        logger.log("Training NNDynamicsModel - finished epoch %i -- train loss: %.4f  valid loss: %.4f  valid_loss_mov_avg: %.4f"%(epoch, float(np.mean(batch_losses)), valid_loss, valid_loss_rolling_average))
                    break

            if early_stopping and valid_loss_rolling_average_prev < valid_loss_rolling_average:
                logger.log('Stopping DynamicsEnsemble Training since valid_loss_rolling_average decreased')
                break
            valid_loss_rolling_average_prev = valid_loss_rolling_average

    def predict(self, obs, act):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :return: pred_obs_next: predicted batch of next observations (n_samples, ndim_obs)
        """
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            inputs = tf.concat([obs, act], axis=1)
            delta = self.f_delta_pred(inputs).numpy()
            delta = denormalize(delta, self._mean_delta_var, self._std_delta_var)
        else:
            inputs = tf.concat([obs, act], axis=1)
            delta = self.f_delta_pred(inputs).numpy()

        pred_obs = obs_original + delta
        next_obs = np.clip(pred_obs, -1e2, 1e2)
        return next_obs


    def predict_sym(self, obs_var, act_var):
        obs_original = obs_var
        in_obs_var = (obs_var - self._mean_obs_var) / (self._std_obs_var + 1e-8)
        in_act_var = (act_var - self._mean_act_var) / (self._std_act_var + 1e-8)
        input_var = tf.concat([in_obs_var, in_act_var], axis=1)
        delta_pred = self.f_delta_pred(input_var) * self._std_delta_var + self._mean_delta_var
        next_obs = obs_original + delta_pred
        next_obs = tf.clip_by_value(next_obs, -1e2, 1e2)
        return next_obs.numpy()
        # log_std = tf.log(self._std_delta_var)
        # return dict(mean=mean, log_std=log_std)

    def compute_normalization(self, obs, act, delta):
        """
        Computes the mean and std of the data and saves it in a instance variable
        -> the computed values are used to normalize the data at train and test time
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after takeing action - numpy array of shape (n_samples, ndim_obs)
        """

        assert obs.shape[0] == delta.shape[0] == act.shape[0]
        #delta = obs_next - obs
        assert delta.ndim == 2 and delta.shape[0] == delta.shape[0]

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization['obs'] = (np.mean(obs, axis=0), np.std(obs, axis=0))
        self.normalization['delta'] = (np.mean(delta, axis=0), np.std(delta, axis=0))
        self.normalization['act'] = (np.mean(act, axis=0), np.std(act, axis=0))

        # sess.run(self._assignations, feed_dict={self._mean_obs_ph: self.normalization['obs'][0],
        #                                         self._std_obs_ph: self.normalization['obs'][1],
        #                                         self._mean_act_ph: self.normalization['act'][0],
        #                                         self._std_act_ph: self.normalization['act'][1],
        #                                         self._mean_delta_ph: self.normalization['delta'][0],
        #                                         self._std_delta_ph: self.normalization['delta'][1],
        #                                         })

        self._mean_obs_var.assign(self.normalization['obs'][0])
        self._std_obs_var.assign(self.normalization['obs'][1])
        self._mean_act_var.assign(self.normalization['act'][0])
        self._std_act_var.assign(self.normalization['act'][1])
        self._mean_delta_var.assign(self.normalization['delta'][0])
        self._std_delta_var.assign(self.normalization['delta'][1])

    def _create_stats_vars(self):
        zero_initializer = tf.keras.initializers.Zeros()
        ones_initializer = tf.keras.initializers.Ones()
        self._mean_obs_var = tf.Variable(name = 'mean_obs',
                                             dtype=tf.float32, initial_value=zero_initializer(shape=(self.obs_space_dims,)), trainable=False)
        self._std_obs_var = tf.Variable(name = 'std_obs',
                                              dtype=tf.float32, initial_value=ones_initializer(shape=(self.obs_space_dims,)), trainable=False)
        self._mean_act_var = tf.Variable(name = 'mean_act',
                                               dtype=tf.float32, initial_value=zero_initializer(shape=(self.action_space_dims,)), trainable=False)
        self._std_act_var = tf.Variable(name = 'std_act',
                                              dtype=tf.float32, initial_value=ones_initializer(shape=(self.action_space_dims,)), trainable=False)
        self._mean_delta_var = tf.Variable(name = 'mean_delta',
                                                dtype=tf.float32, initial_value=zero_initializer(shape=(self.obs_space_dims,)), trainable=False)
        self._std_delta_var = tf.Variable(name = 'std_delta',
                                               dtype=tf.float32, initial_value=ones_initializer(shape=(self.obs_space_dims,)), trainable=False)

        #self._mean_obs_ph = tf.placeholder(tf.float32, shape=(self.obs_space_dims,))
        #self._std_obs_ph = tf.placeholder(tf.float32, shape=(self.obs_space_dims,))
        #self._mean_act_ph = tf.placeholder(tf.float32, shape=(self.action_space_dims,))
        #self._std_act_ph = tf.placeholder(tf.float32, shape=(self.action_space_dims,))
        #self._mean_delta_ph = tf.placeholder(tf.float32, shape=(self.obs_space_dims,))
        #self._std_delta_ph = tf.placeholder(tf.float32, shape=(self.obs_space_dims,))

        # self._assignations = [tf.assign(self._mean_obs_var, self._mean_obs_ph),
        #                       tf.assign(self._std_obs_var, self._std_obs_ph),
        #                       tf.assign(self._mean_act_var, self._mean_act_ph),
        #                       tf.assign(self._std_act_var, self._std_act_ph),
        #                       tf.assign(self._mean_delta_var, self._mean_delta_ph),
        #                       tf.assign(self._std_delta_var, self._std_delta_ph),
        #                       ]

    def _data_input_fn(self, obs, act, delta, batch_size=500, buffer_size=5000):
        """ Takes in train data an creates an a symbolic nex_batch operator as well as an iterator object """

        assert obs.ndim == act.ndim == delta.ndim, "inputs must have 2 dims"
        assert obs.shape[0] == act.shape[0] == delta.shape[0], "inputs must have same length along axis 0"
        assert obs.shape[1] == delta.shape[1], "obs and obs_next must have same length along axis 1 "

        #self.obs_dataset_ph = tf.placeholder(tf.float32, (None, obs.shape[1]))
        #self.act_dataset_ph = tf.placeholder(tf.float32, (None, act.shape[1]))
        #self.delta_dataset_ph = tf.placeholder(tf.float32, (None, delta.shape[1]))

        dataset = tf.data.Dataset.from_tensor_slices(
            (obs, act, delta)
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        return dataset

    def _normalize_data(self, obs, act, delta=None):
        """normalize data"""
        """ YOUR CODE HERE FOR PROBLEM 2A.3"""
        # hint: you should write your own normalize function in hw5.dynamics.utils.normalize, and utilize it here!
        # self.normalization contains mean and std. e.g. self.normalization['obs'][0] is the mean of obs.
        obs_normalized = normalize(obs, self._mean_obs_var, self._std_obs_var)
        actions_normalized = normalize(act, self._mean_act_var, self._std_act_var)

        if delta is not None:
            deltas_normalized = normalize(delta, self._mean_delta_var, self._std_delta_var)
            """ YOUR CODE ENDS"""
            return obs_normalized, actions_normalized, deltas_normalized
        else:
            return obs_normalized, actions_normalized


    def initialize_unitialized_variables(self, sess):
        uninit_variables = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_variables.append(var)

        sess.run(tf.variables_initializer(uninit_variables))

    def reinit_model(self):
        sess = tf.get_default_session()
        if '_reinit_model_op' not in dir(self):
            self._reinit_model_op = [tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope=self.name + '/dynamics_model'))]
        sess.run(self._reinit_model_op)

    def __getstate__(self):
        # state = LayersPowered.__getstate__(self)
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        state['normalization'] = self.normalization
        state['networks'] = [nn.__getstate__() for nn in self._networks]
        return state

    def __setstate__(self, state):
        # LayersPowered.__setstate__(self, state)
        Serializable.__setstate__(self, state['init_args'])
        self.normalization = state['normalization']
        for i in range(len(self._networks)):
            self._networks[i].__setstate__(state['networks'][i])

