import numpy as np
from gym import spaces


class LookAheadPolicy(object):
    """
    Look ahead policy

    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
    * self.horizon (int): Horizon for the look ahead policy

    * act_dim (int): Dimension of the state space

    * value_fun (TabularValueFun):
                - get_values(states): if states is None returns the values of all the states. Otherwise, it returns the
                                      values of the specified states
    * env (Env):
                - vec_set_state(states): vectorized (multiple environments in parallel) version of reseting the
                environment to a state for a batch of states.
                - vec_step(actions): vectorized (multiple environments in parallel) version of stepping through the
                environment for a batch of actions. Returns the next observations, rewards, dones signals, env infos
                (last not used).
    """
    def __init__(self,
                 env,
                 value_fun,
                 horizon,
                 ):
        self.env = env
        self.discount = env.discount
        self._value_fun = value_fun
        self.horizon = horizon

    def get_action(self, state):
        """
        Get the best action by doing look ahead, covering actions for the specified horizon.
        HINT: use np.meshgrid to compute all the possible action sequences.
        :param state:
        :return: best_action (int)
           """

        assert isinstance(self.env.action_space, spaces.Discrete)
        act_n = self.env.action_space.n
        """ INSERT YOUR CODE HERE"""
        action_combinations = np.repeat(np.arange(act_n).reshape(-1, act_n), self.horizon, axis=0)
        action_sequences = np.array(np.meshgrid(*action_combinations)).T.reshape(self.horizon, -1)

        returns = self.get_returns(state, action_sequences) # shape = (action_sequences.shape[1],)
        best_action = action_sequences[0, np.argmax(returns)]


        return best_action

    def get_returns(self, state, actions):
        """
        :param state: current state of the policy
        :param actions: array of actions of shape [horizon, num_acts]
        :return: returns for the specified horizon + self.discount ^ H value_fun
        HINT: Make sure to take the discounting and done into acount!
        """
        assert self.env.vectorized
        """ INSERT YOUR CODE HERE"""

        returns = np.zeros(shape=(actions.shape[1]))
        next_state_batch = np.repeat(state, actions.shape[1])
        for i in range(self.horizon):
            action_batch = actions[i,:]
            state_batch = next_state_batch
            self.env.vec_set_state(state_batch)
            next_state_batch, rewards, done, _ = self.env.vec_step(action_batch)
            next_state_batch = done * self.env.obs_n + (~done * next_state_batch)
            returns += self.discount ** i * rewards

        trans_probs = self.env.transitions[state_batch, action_batch]
        trans_idx = self.env.transitions._idxs[state_batch, action_batch]
        returns += self.discount ** self.horizon * np.sum(self._value_fun.get_values(trans_idx) * trans_probs, axis=1)

        #returns += self.discount ** self.horizon * self._value_fun.get_values(next_state_batch)
        return returns

    def update(self, actions):
        pass
