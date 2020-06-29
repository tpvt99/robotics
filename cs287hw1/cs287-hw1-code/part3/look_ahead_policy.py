import numpy as np
from gym import spaces
from part2.look_ahead_policy import LookAheadPolicy as BaseLookAheadPolicy


class LookAheadPolicy(BaseLookAheadPolicy):
    """
    Look ahead policy

    -- UTILS VARIABLES FOR RUNNING THE CODE --
    * look_ahead_type (str): Type of look ahead policy to use

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
    * self.horizon (int): Horizon for the look ahead policy

    * act_dim (int): Dimension of the state space

    * self.num_elites (int): number of best actions to pick for the cross-entropy method

    * self.value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states

    * self.get_returns_state(state): It is the same that you implemented in the previous part
    """
    def __init__(self,
                 env,
                 value_fun,
                 horizon,
                 look_ahead_type='tabular',
                 num_acts=20,
                 cem_itrs=10,
                 precent_elites=0.25,
                 ):
        self.env = env
        self.discount = self.env.discount
        self._value_fun = value_fun
        self.horizon = horizon
        self.num_acts = num_acts
        self.cem_itrs = cem_itrs
        self.num_elites = int(num_acts * precent_elites)
        assert self.num_elites > 0
        self.look_ahead_type = look_ahead_type

    def get_action(self, state):
        if self.look_ahead_type == 'tabular':
            action = self.get_action_tabular(state)
        elif self.look_ahead_type == 'rs':
            action = self.get_action_rs(state)
        elif self.look_ahead_type == 'cem':
            action = self.get_action_cem(state)
        else:
            raise NotImplementedError
        return action

    def get_action_cem(self, state):
        """
        Do lookahead in the continous and discrete case with the cross-entropy method..
        :param state (int if discrete np.ndarray if continous)
        :return: best_action (int if discrete np.ndarray if continous)
        """
        num_acts = self.num_acts

        if isinstance(self.env.observation_space, spaces.Discrete):
            # shape = np.array(num_acts,), each state is only 1 dimension whose numbers are integers
            state_sequences = np.tile(state, num_acts)
        else:
            assert num_acts is not None
            # shape = [num_acts, state_dims]
            state_sequences = np.tile(state, (num_acts,1))

        """ INSERT YOUR CODE HERE"""
        if isinstance(self.env.action_space, spaces.Discrete):
            action_sequences = np.array([self.env.action_space.sample() \
                                         for i in range(self.horizon * num_acts)]).reshape(self.horizon, num_acts)


            for _ in range(self.cem_itrs):
                action_counts = np.zeros(shape=self.env.action_space.n)
                returns = self.get_returns(state_sequences, action_sequences)
                elites_idx = np.argsort(returns)[-self.num_elites:]
                for i in action_sequences[0][elites_idx]:
                    action_counts[i] += 1
                action_sequences = np.array([np.random.choice(self.env.action_space.n, num_acts, p = action_counts / np.sum(action_counts))\
                                             for i in range(self.horizon)]).reshape(self.horizon, num_acts)
            best_action = np.random.choice(self.env.action_space.n, 1, p = action_counts / np.sum(action_counts))[0]

        else:
            sigma_of_actions = np.ones(shape=self.env.action_space.shape[0])
            mu_of_actions = np.array([np.random.normal(0, sigma_of_actions)
                                      for i in range(self.horizon * self.num_acts)]).reshape(self.horizon, num_acts, self.env.action_space.shape[0])

            for _ in range(self.cem_itrs):
                returns = self.get_returns(state_sequences, mu_of_actions)
                elites_idx = np.argsort(returns)[-self.num_elites:]
                elites_actions_mean = mu_of_actions[0, elites_idx, :].mean(axis=0)
                elites_actions_std = mu_of_actions[0, elites_idx, :].std(axis=0)

                mu_of_actions = np.array([np.random.normal(elites_actions_mean, elites_actions_std)
                                      for i in range(self.horizon * self.num_acts)]).reshape(self.horizon, num_acts, self.env.action_space.shape[0])


            best_action = elites_actions_mean


        return best_action

    def get_action_rs(self, state):
        """
        Do lookahead in the continous and discrete case with random shooting..
        :param state (int if discrete np.ndarray if continous)
        :return: best_action (int if discrete np.ndarray if continous)
        """
        num_acts = self.num_acts
        """ INSERT YOUR CODE HERE """
        if isinstance(self.env.action_space, spaces.Discrete):
            # shape = [self.horizon, num_acts], each action is only 1 dimension whose numbers are integers
            action_sequences = np.array([self.env.action_space.sample() \
                                         for i in range(self.horizon * num_acts)]).reshape(self.horizon, num_acts)
        else:
            assert num_acts is not None
            # shape = [self.horizon, num_acts, acts_dims], numbers are floating points
            action_sequences = np.array([self.env.action_space.sample() \
                                         for i in range(self.horizon * num_acts)]).reshape(self.horizon,
                                         num_acts, self.env.action_space.shape[0])

        if isinstance(self.env.observation_space, spaces.Discrete):
            # shape = np.array(num_acts,), each state is only 1 dimension whose numbers are integers
            state_sequences = np.tile(state, num_acts)
        else:
            assert num_acts is not None
            # shape = [num_acts, state_dims]
            state_sequences = np.tile(state, (num_acts,1))

        returns = self.get_returns(state_sequences, action_sequences)

        best_action = action_sequences[0, np.argmax(returns)]
        return best_action

    def get_returns(self, states, actions):
        """
        :param states: current states of the policy of shape [num_acts, states_dims] if continuous; shape [num_acts] if discrete
        :param actions: array of actions of shape [horizon, num_acts, acts_dims] if continuous; shape [horizon, num_acts] if discrete
        :return: returns for the specified horizon + self.discount ^ H value_fun
        HINT: Make sure to take the discounting and done into acount!
        """
        assert self.env.vectorized

        returns = np.zeros(shape=(actions.shape[1]))
        next_state_batch = states

        for i in range(self.horizon):
            action_batch = actions[i,:]
            state_batch = next_state_batch
            self.env.vec_set_state(state_batch)
            next_state_batch, rewards, done, _ = self.env.vec_step(action_batch)
            returns += self.discount ** i * rewards

        returns += self.discount ** self.horizon * self._value_fun.get_values(next_state_batch)
        return returns
