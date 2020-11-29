import tensorflow as tf
import time

class SAC:
    """Soft Actor-Critic (SAC)
    Original code from Tuomas Haarnoja, Soroush Nasiriany, and Aurick Zhou for CS294-112 Fall 2018

    References
    ----------
    [1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
        Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor," ICML 2018.
    """

    def __init__(self,
                 alpha=1.0,
                 batch_size=256,
                 discount=0.99,
                 epoch_length=1000,
                 learning_rate=3e-3,
                 reparameterize=False,
                 tau=0.01,
                 **kwargs):
        """
        Args:
        """

        self._alpha = alpha
        self._batch_size = batch_size
        self._discount = discount
        self._epoch_length = epoch_length
        self._learning_rate = learning_rate
        self._reparameterize = reparameterize
        self._tau = tau

        self._training_ops = []

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

    def build(self, env, policy, q_function, q_function2, value_function,
              target_value_function):

        self.env = env
        self.policy = policy
        self.q_function = q_function
        self.q_function2 = q_function2
        self.value_function = value_function
        self.target_value_function = target_value_function


    def run(self, data):
        #optimize Value function
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.value_function.trainable_variables)
            value_function_loss = self._value_function_loss_for(data)
        gradients = tape.gradient(value_function_loss, self.value_function.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.value_function.trainable_variables))

        #optimizer q function
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.q_function.trainable_variables)
            q_loss = self._q_function_loss_for(self.q_function, data)
        gradients = tape.gradient(q_loss, self.q_function.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_function.trainable_variables))

        if self.q_function2 is not None:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.q_function2.trainable_variables)
                q_loss = self._q_function_loss_for(self.q_function2, data)
            gradients = tape.gradient(q_loss, self.q_function2.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.q_function2.trainable_variables))

        # optimize policy
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.policy.trainable_variables)
            policy_loss = self._policy_loss_for(data)
        gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

        self.target_update(self.target_value_function, self.value_function)


    def _policy_loss_for(self, batch_data):
        obs, act, next_obs, rewards, done = batch_data['observations'], batch_data['actions'], \
                    batch_data['next_observations'], batch_data['rewards'], batch_data['terminals']

        if not self._reparameterize:
            """  CODE PROVIDED """

            actions, log_pis = self.policy(obs)
            if self.q_function2 is None:
                q_vals = self.q_function([obs, actions])
            else:
                """ YOUR CODE HERE FOR PROBLEM 3A.3"""
                q_vals = tf.math.minimum(
                    self.q_function([obs, actions]),
                    self.q_function2([obs, actions])
                )
            q_vals = tf.squeeze(q_vals, axis=1)
                
            baseline = self.value_function(obs)
            baseline = tf.squeeze(baseline, axis=1)
            
            target = self._alpha * log_pis - q_vals + baseline
            
            target = tf.stop_gradient(target)
            result = tf.reduce_mean(log_pis * target)
            """ CODE ENDS """
            return result
        else:
            """ CODE PROVIDED"""
            actions, log_pis = self.policy(obs)
            if self.q_function2 is None:
                q_vals = self.q_function([obs, actions])
            else:
                q_vals = tf.math.minimum(
                    self.q_function([obs, actions]),
                    self.q_function2([obs, actions])
                )
            q_vals = tf.squeeze(q_vals, axis=1)    
            
            result = tf.reduce_mean(
                self._alpha * log_pis - q_vals
            )
            """ CODE ENDS """
            return result

    def _value_function_loss_for(self, batch_data):
        """
        :param policy:
        :param q_function:
        :param q_function2:
        :param value_function:
        :return:
        """
        """ YOUR CODE HERE FOR PROBLEM 3A.2"""

        obs, act, next_obs, rewards, done = batch_data['observations'], batch_data['actions'], \
                            batch_data['next_observations'], batch_data['rewards'], batch_data['terminals']

        if self.q_function2 is None:
            v_val = self.value_function(obs)
            next_act, log_pi = self.policy(obs)
            q_val = self.q_function([obs, next_act])
            loss = v_val - q_val + self._alpha * log_pi

        else:
            """ YOUR CODE HERE FOR PROBLEM 3A.3"""
            v_val = self.value_function(obs)
            next_act, log_pi = self.policy(obs)
            q_val1 = self.q_function([obs, next_act])
            q_val2 = self.q_function2([obs, next_act])

            loss = v_val - tf.math.minimum(q_val1, q_val2) + self._alpha * log_pi
        return loss


    def _q_function_loss_for(self, q_function, batch_data):
        """ q loss """
        """ YOUR CODE HERE FOR PROBLEM 3A.1"""
        obs, act, next_obs, rewards, done = batch_data['observations'], batch_data['actions'], \
                        batch_data['next_observations'], batch_data['rewards'], batch_data['terminals']

        q_val = q_function([obs, act])
        v_val = self.target_value_function(next_obs)
        return tf.reduce_mean((q_val - (rewards + self._alpha * v_val))**2)

    def target_update(self, source, target):
        """Create tensorflow operations for updating target value function."""

        for target, source in zip(target.trainable_variables, source.trainable_variables):
            target.assign((1 - self._tau) * target + self._tau * source)

    def train(self, sampler, n_epochs=1000):
        """Return a generator that performs RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        self._start = time.time()
        for epoch in range(n_epochs):
            for t in range(self._epoch_length):
                sampler.sample()

                batch = sampler.random_batch(self._batch_size)
                self.run(batch)

            yield epoch

    def get_statistics(self):
        statistics = {
            'Time': time.time() - self._start,
            'TimestepsThisBatch': self._epoch_length,
        }

        return statistics
