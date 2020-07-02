import numpy as np
from utils.utils import DiscretizeWrapper


class Discretize(DiscretizeWrapper):
    """
    Discretize class: Discretizes a continous gym environment


    -- VARIABLES/FUNCTIONS YOU WILL NEED TO USE --
        * self.state_points (np.ndarray): grid that contains the real values of the discretization

        * self.obs_n (int): number of discrete points

        * self.transitions (np.ndarray): transition matrix of size (S+1, A, S+1). The last state corresponds to the sink
                                         state
        * self.rewards (np.ndarray): reward matrix of size (S+1, A, S+1). The last state corresponds to the sink state

        * self.get_id_from_coordinates(coordinate_vector) returns the id of the coordinate_vector

        * self.get_state_from_id(id_s): get the continuous state associated to that state id

        * self.get_action_from_id(id_a): get the contiouns action associated to that action id

        * env.set_state(s): resets the environment to the continous state s

        * env.step(a): applies the action a to the environment. Returns next_state, reward, done, env_infos. The
                            last element is not used.
    """

    def vec_get_discrete_state_from_cont_state(self, cont_states):
        """
        Get discrete state from continuous state.
            * self.mode (str): specifies if the discretization is to the nearest-neighbour (nn) or n-linear (linear).

        :param cont_state (np.ndarray): of shape env.observation_space.shape
        :return: A tuple of (states, probs). states is a np.ndarray of shape (max_points,) if mode=='nn' (max_points are the 1st dims of cont_states)
                and (max_points, 2 ^ obs_dim) if mode=='linear'. probs is the probabability of going to such states,
                it has the same size than states.
        """
        """INSERT YOUR CODE HERE"""
        cont_states = np.expand_dims(cont_states, axis=-1) # expands dims to be able to subtract from self.state_points
        if self.mode == 'nn':
            if cont_states.ndim == 3: # dims 3 when called from vectorization version
                coordinates = np.abs(cont_states - self.state_points).argmin(axis=2)
            elif cont_states.ndim == 2: # dims 2 when called in each run time step
                coordinates = np.abs(cont_states - self.state_points).argmin(axis=1)

            states = self.get_id_from_coordinates(coordinates)

            if cont_states.ndim == 2: # dims 2 then states are single integer -> must convert to array
                states = np.array([states])
            probs = np.ones_like(states)


        elif self.mode == 'linear':

            obs_dim = self.obs_dims
            min_val = np.abs(cont_states - self.state_points)

            if cont_states.ndim == 3:
                nearest_coordinates = np.argsort(min_val)[:,:,
                                      0:2]  # first row is the nearest and 2nd-nearest to 1st dims of obs_dims,
                # second tow is the nearest and 2nd-nearest to 2nd dims of obs_dim, ... continue to obs_dims
                nearest_probs = np.sort(min_val, axis=2)[:, :, 0:2]

                coordinates_combinations = np.stack([np.array(np.meshgrid(*nearest_coordinates[i,:])).T.reshape(-1, self.obs_dims)
                                                    for i in range(cont_states.shape[0])]).reshape(cont_states.shape[0], -1, self.obs_dims)

                probs_combinations = np.stack([np.array(np.meshgrid(*nearest_probs[i,:])).T.reshape(-1, self.obs_dims)
                                                    for i in range(cont_states.shape[0])]).reshape(cont_states.shape[0], -1, self.obs_dims)

                probs_combinations = np.prod(probs_combinations, axis=2)
                probs_combinations = 1. / (probs_combinations + 1e-9)  # the smaller probs mean it is higher probs to get to that state. so we have to reverse
                probs_combinations = probs_combinations/ np.expand_dims(np.sum(probs_combinations, axis=1), axis=-1)  # do the normalization
                states = self.get_id_from_coordinates(coordinates_combinations)
                probs = probs_combinations

            elif cont_states.ndim == 2:
                nearest_coordinates = np.argsort(min_val)[:,
                                      0:2]  # first row is the nearest and 2nd-nearest to 1st dims of obs_dims,
                # second tow is the nearest and 2nd-nearest to 2nd dims of obs_dim, ... continue to obs_dims
                nearest_probs = np.sort(min_val)[:, 0:2]

                coordinates_combinations = np.array(np.meshgrid(*nearest_coordinates)).T.reshape(-1, self.obs_dims)
                probs_combinations = np.array(np.meshgrid(*nearest_probs)).T.reshape(-1, self.obs_dims)
                probs_combinations = np.prod(probs_combinations, axis=1)
                probs_combinations = 1. / (
                            probs_combinations + 1e-9)  # the smaller probs mean it is higher probs to get to that state. so we have to reverse
                probs_combinations = probs_combinations / np.sum(probs_combinations)  # do the normalization

                states = self.get_id_from_coordinates(coordinates_combinations)
                probs = probs_combinations
        else:
            raise NotImplementedError
        return states, probs

    def vec_add_transition(self, id_s, id_a):
        """
        Populates transition and reward matrix (self.transition and self.reward)
        :param id_s (np.array): discrete index array of the the states
        :param id_a (np.array): discrete index array of the the actions

        """
        env = self._wrapped_env
        obs_n = self.obs_n

        """INSERT YOUR CODE HERE"""
        cont_states = self.get_state_from_id(id_s)
        cont_actions = self.get_action_from_id(id_a)
        env.vec_set_state(cont_states)
        next_cont_states, rewards, dones, _ = env.vec_step(cont_actions)
        next_disc_states, probs = self.vec_get_discrete_state_from_cont_state(next_cont_states)

        if self.mode == 'nn':
            self.transitions[id_s, id_a, next_disc_states] = probs
            self.rewards[id_s, id_a, next_disc_states] = rewards
        elif self.mode == 'linear':
            self.transitions[id_s, id_a, next_disc_states] = probs
            self.rewards[id_s, id_a, next_disc_states] = rewards

    def add_done_transitions(self):
        """
        Populates transition and reward matrix for the sink state (self.transition and self.reward). The sink state
        corresponds to the last state (self.obs_n or -1).
        """
        """INSERT YOUR CODE HERE"""
        self.transitions[self.obs_n, :, 0] = 1 # when done, reset to the state 0 as stated in the env.py files
        self.rewards[self.obs_n, :, 0] = 0 # when done, set rewards = 0 as stated in the env.py files



