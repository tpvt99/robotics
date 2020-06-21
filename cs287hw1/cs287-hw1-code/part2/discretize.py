import numpy as np
from utils.utils import DiscretizeWrapper


class Discretize(DiscretizeWrapper):
    """
    Discretize class: Discretizes a continous gym environment


    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
        * self.state_points (np.ndarray): grid that contains the real values of the discretization

        * self.obs_n (int): number of discrete points

        * self.transitions (np.ndarray): transition matrix of size (S+1, A, n_obs_next). The last state corresponds to the sink
                                         state. n_obs_next depend on the mode is nearest-neighbor or linear-interpolation
        * self.rewards (np.ndarray): reward matrix of size (S+1, A, n_obs_next). The last state corresponds to the sink state

        * self.get_id_from_coordinates(coordinate_vector) returns the id of the coordinate_vector

        * self.get_state_from_id(id_s): get the discretized continuous state associated to that state id

        * self.get_action_from_id(id_a): get the discretized continuous action associated to that action id

        * env.set_state(s): resets the environment to the continuous state s

        * env.step(a): applies the action a to the environment. Returns next_state, reward, done, env_infos. The
                            last element is not used.
    """

    def get_discrete_state_from_cont_state(self, cont_state):
        """
        Get discrete state from continuous state.
            * self.mode (str): specifies if the discretization is to the nearest-neighbour (nn) or n-linear (linear).

        :param cont_state (np.ndarray): of shape env.observation_space.shape
        :return: A tuple of (states, probs). states is a np.ndarray of shape (1,) if mode=='nn'
                and (2 ^ obs_dim,) if mode=='linear'. probs is the probability of going to such states,
                it has the same size as states.
        """
        """INSERT YOUR CODE HERE"""
        cont_state = np.expand_dims(cont_state, axis=-1)
        if self.mode == 'nn':
            obs_dim = self.obs_dims
            bin_per_dim = (int) (self.obs_n **(1/obs_dim))
            coordinates = np.zeros_like(cont_state)
            for i in range(obs_dim):
                for bin in range(bin_per_dim-1):
                    if cont_state[i] >= self.state_points[i][bin] and cont_state[i] < self.state_points[i][bin+1]:
                        if abs(cont_state[i] - self.state_points[i][bin]) <= abs(cont_state[i] - self.state_points[i][bin+1]):
                            coordinates[i] = bin
                        else:
                            coordinates[i] = bin+1
                        break
                if cont_state[i] >= self.state_points[i][-1]: # cater for the last point
                    coordinates[i] = bin_per_dim-1
                elif cont_state[i] <= self.state_points[i][0]:
                    coordinates[i] = 0
            states = self.get_id_from_coordinates(coordinates.flatten())
            states = np.array([states])
            probs = np.array([1.0])

        elif self.mode == 'linear':
            raise NotImplementedError
            """Your code ends here"""
        else:
            raise NotImplementedError
        return states, probs

    def add_transition(self, id_s, id_a):
        """
        Populates transition and reward matrix (self.transition and self.reward)
        :param id_s (int): discrete index of the the state
        :param id_a (int): discrete index of the the action

        """
        env = self._wrapped_env
        obs_n = self.obs_n

        """INSERT YOUR CODE HERE"""
        cont_action = self.get_action_from_id(id_a)
        cont_state = self.get_state_from_id(id_s)
        env.set_state(cont_state)
        next_cont_state, reward, done, env_info = env.step(cont_action)
        next_dis_state, probs = self.get_discrete_state_from_cont_state(next_cont_state)
        if done:
            self.transitions[id_s, id_a, obs_n] = np.array([1])
            self.rewards[id_s, id_a, obs_n] = reward
        else:

            self.transitions[id_s, id_a, next_dis_state] = probs
            self.rewards[id_s, id_a, next_dis_state] = reward
    def add_done_transitions(self):
        """
        Populates transition and reward matrix for the sink state (self.transition and self.reward). The sink state
        corresponds to the last state (self.obs_n or -1).
        """
        """INSERT YOUR CODE HERE"""
        self.transitions[self.obs_n, :, self.obs_n] = np.array([1])
        self.rewards[self.obs_n, :, self.obs_n] = np.array([0])


