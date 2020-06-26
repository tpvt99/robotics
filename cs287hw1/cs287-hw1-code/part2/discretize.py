import numpy as np
from utils.utils import DiscretizeWrapper
import math
import itertools

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
            coordinates = np.abs(cont_state - self.state_points).argmin(axis=1)
            states = self.get_id_from_coordinates(coordinates)
            states = np.array([states])
            probs = np.array([1.0])

        elif self.mode == 'linear':
            obs_dim = self.obs_dims
            min_val = np.abs(cont_state - self.state_points)

            nearest_coordinates = np.argsort(min_val)[:, 0:2] # first row is the nearest and 2nd-nearest to 1st dims of obs_dims,
                                                                # second tow is the nearest and 2nd-nearest to 2nd dims of obs_dim, ... continue to obs_dims
            nearest_probs = np.sort(min_val)[:, 0:2]

            coordinates_combinations = np.array(np.meshgrid(*nearest_coordinates)).T.reshape(-1, self.obs_dims)
            probs_combinations = np.array(np.meshgrid(*nearest_probs)).T.reshape(-1, self.obs_dims)
            probs_combinations = np.prod(probs_combinations, axis=1)
            probs_combinations = 1./(probs_combinations + 1e-9) # the smaller probs mean it is higher probs to get to that state. so we have to reverse
            probs_combinations = probs_combinations / np.sum(probs_combinations) # do the normalization

            states = self.get_id_from_coordinates(coordinates_combinations)
            probs = probs_combinations

            # code works
            # astates = np.zeros(shape=(2**self.obs_dims))
            # aprobs = np.zeros(shape=(2**self.obs_dims))
            # for i in range(2**self.obs_dims):
            #     new_coordinates = np.zeros(shape=(self.obs_dims))
            #     p = 1
            #     z=i
            #     for dim in range(self.obs_dims):
            #         bit = (int) (z%2)
            #         new_coordinates[dim] = nearest_coordinates[dim, bit]
            #         p = p * nearest_probs[dim, bit]
            #         z = z/2
            #     astates[i] = self.get_id_from_coordinates(new_coordinates)
            #     aprobs[i] = 1./(p + 1e-9)
            # aprobs = aprobs / np.sum(aprobs) # normalized
            # astates = astates.astype(int)
            # assert (aprobs >= 0).all()


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
            if self.mode == 'nn':
                self.transitions[id_s, id_a, next_dis_state] = probs
                self.rewards[id_s, id_a, next_dis_state] = reward
            elif self.mode == 'linear':
                for i, n_obs in enumerate(next_dis_state):
                    self.transitions[id_s, id_a, n_obs] = probs[i]
                    self.rewards[id_s, id_a, n_obs] = reward

    def add_done_transitions(self):
        """
        Populates transition and reward matrix for the sink state (self.transition and self.reward). The sink state
        corresponds to the last state (self.obs_n or -1).
        """
        """INSERT YOUR CODE HERE"""
        self.transitions[self.obs_n, :, self.obs_n] = np.array([1])
        self.rewards[self.obs_n, :, self.obs_n] = np.array([0])


