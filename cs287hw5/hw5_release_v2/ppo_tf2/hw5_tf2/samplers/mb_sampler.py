from hw5_tf2.samplers.base import Sampler
from hw5_tf2.utils import utils
from hw5_tf2.logger import logger
import numpy as np
import tensorflow as tf
from collections import OrderedDict


class MBSampler(Sampler):

    def __init__(
            self,
            env,
            policy,
            dynamics_model,
            num_rollouts,
            max_path_length,
            parallel=False,
            deterministic_policy=False,
            optimize_actions=False,
            **kwargs
    ):
        super(MBSampler, self).__init__(env, policy, num_rollouts, max_path_length)
        assert not parallel

        self.env = env
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.max_path_length = max_path_length
        self.total_samples = num_rollouts * max_path_length
        self.num_rollouts = num_rollouts
        self.total_timesteps_sampled = 0
        self.deterministic_policy = deterministic_policy
        self.optimize_actions = optimize_actions
        self.num_models = getattr(dynamics_model, 'num_models', 1)


    def samples(self, initial_obs):
        obses = []
        acts = []
        rewards = []
        means = []
        log_stds = []
        obs = initial_obs
        for t in range(self.max_path_length):
            dist_policy = self.policy.distribution_info_sym(obs)
            act, dist_policy = self.policy.distribution.sample_sym(dist_policy)
            next_obs = self.dynamics_model.predict_sym(obs, act)

            reward = self.env.tf_reward(obs, act, next_obs)

            obses.append(obs)
            acts.append(act)
            rewards.append(reward)
            means.append(dist_policy['mean'])
            log_stds.append(dist_policy['log_std'])

            obs = next_obs

        returns_var = tf.reduce_sum(rewards, axis=0)
        rewards_var = rewards
        actions_var = acts
        observations_var = obses
        means_var = means
        log_stds_var = log_stds

        return observations_var, actions_var, means_var, log_stds_var, rewards_var

    def obtain_samples(self, log=False, log_prefix='', buffer=None, random=False):
        """
        Collect batch_size trajectories from each task
        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger
            random (boolean): whether the actions are random
        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        policy = self.policy
        policy.reset(dones=[True] * self.num_rollouts)

        # initial reset of meta_envs
        init_obses = np.array([self.env.reset() for _ in range(self.num_rollouts)])

        observations, actions, means, log_stds, rewards = self.samples(init_obses)

        means = np.array(means).transpose((1, 0, 2))
        log_stds = np.array(log_stds).transpose((1, 0, 2))
        if log_stds.shape[0] == 1:
            log_stds = np.repeat(log_stds, self.num_rollouts, axis=0)
        agent_infos = [dict(mean=mean, log_std=log_std) for mean, log_std in zip(means, log_stds)]
        observations = np.array(observations).transpose((1, 0, 2))
        actions = np.array(actions).transpose((1, 0, 2))
        rewards = np.array(rewards).T
        dones = [[False for _ in range(self.max_path_length)] for _ in range(self.num_rollouts)]
        env_infos = [dict() for _ in range(self.num_rollouts)]
        paths = [dict(observations=obs, actions=act, rewards=rew,
                      dones=done, env_infos=env_info, agent_infos=agent_info) for
                 obs, act, rew, done, env_info, agent_info in
                 zip(observations, actions, rewards, dones, env_infos, agent_infos)]
        self.total_timesteps_sampled += self.total_samples
        logger.logkv('ModelSampler-n_timesteps', self.total_timesteps_sampled)

        return paths

