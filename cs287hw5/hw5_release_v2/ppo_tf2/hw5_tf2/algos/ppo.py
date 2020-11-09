from hw5_tf2.logger import logger
from hw5_tf2.algos.base import Algo
from hw5_tf2.optimizers.first_order_optimizer import FirstOrderOptimizer
from hw5_tf2.utils.serializable import Serializable

import tensorflow as tf
from collections import OrderedDict


class PPO(Algo, Serializable):
    """
    Algorithm for PPO
    policy: neural net that can get actions based on observations
    learning_rate: learning rate for optimization
    clip_eps: 1 + clip_eps and 1 - clip_eps is the range you clip the objective
    max_epochs: max number of epochs
    entropy_bonus: weight on the entropy to encourage exploration
    use_entropy: enable entropy
    use_ppo_obj: enable the ppo_legacy objective instead of policy gradient
    use_clipper: enable clipping the objective
    """
    def __init__(
            self,
            policy,
            name="ppo_legacy",
            learning_rate=1e-3,
            clip_eps=0.2,
            max_epochs=5,
            entropy_bonus=0.,
            use_entropy=True,
            use_ppo_obj=True,
            use_clipper=True,
            **kwargs
            ):
        Serializable.quick_init(self, locals())
        super(PPO, self).__init__(policy)

        # Step 1. Initialize optimizer
        self.optimizer = FirstOrderOptimizer(learning_rate=learning_rate, max_epochs=max_epochs)

        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self._clip_eps = clip_eps
        self.entropy_bonus = entropy_bonus
        self.use_entropy = use_entropy
        self.use_ppo_obj = use_ppo_obj
        self.use_clipper = use_clipper
        self.build_policy()

    def loss_objective(self, data):
        '''

        :param observation: observation has shape (None, obs_dims)
        :param action: action has shape (None, act_dims)
        :return:
        '''

        observations, actions, rewards, returns = data['observations'], data['actions'], data['rewards'], data['returns']

        #1. Find the action_mean and action_log_std by running policy (it is prediction step)
        # distribution_info is dict has 2 keys {'mean': mean, 'log_std': log_std}
        distribution_info = self.policy.distribution_info_sym(observations)

        #2. Find the loss between true_y (action) and predicted_y (distribution_info)
        if self.use_ppo_obj:
            pass
        else:
            loglikelihood = self.policy.distribution.log_likelihood_sym(actions, distribution_info)
            likelihood_ratio = loglikelihood

        if self.use_clipper:
            pass
        else:
            clipped_obj = likelihood_ratio

        if self.use_entropy:
            pass
        else:
            surr_obj = - tf.math.reduce_mean(clipped_obj * returns)

        return surr_obj

    def build_policy(self):

        """ policy gradient objective """
        self.optimizer.build_optimizer(
            loss_object = self.loss_objective,
            target = self.policy
        )


    def optimize_policy(self, samples_data, log=True, prefix='', verbose=False):
        """
        Performs policy optimization

        Args:
            samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update
            log (bool) : whether to log statistics

        Returns:
            None
        """
        #input_dict = self._extract_input_dict(samples_data, self._optimization_keys, prefix='train')

        if verbose:
            logger.log("Optimizing")
        loss_before = self.optimizer.optimize(data=samples_data)

        if verbose:
            logger.log("Computing statistics")
        loss_after = self.optimizer.loss(data=samples_data)

        if log:
            logger.logkv(prefix+'LossBefore', loss_before.numpy())
            logger.logkv(prefix+'LossAfter', loss_after.numpy())

    # def __getstate__(self):
    #     state = dict()
    #     state['init_args'] = Serializable.__getstate__(self)
    #     print('getstate\n')
    #     print(state['init_args'])
    #     state['policy'] = self.policy.__getstate__()
    #     state['optimizer'] = self.optimizer.__getstate__()
    #     return state
    #
    # def __setstate__(self, state):
    #     Serializable.__setstate__(self, state['init_args'])
    #     self.policy.__setstate__(state['policy'])
    #     self.optimizer.__getstate__(state['optimizer'])
