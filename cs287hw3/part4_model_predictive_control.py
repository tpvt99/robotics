import numpy as np
from utils import *
import copy

class MPCPolicy(object):
    def __init__(self, env, horizon):
        self.env = env
        self.H = horizon
        self.env = copy.deepcopy(env)
        np.random.seed(1)
        self.init_actions = np.random.uniform(low=-.1, high=.1, size=(horizon * env.du,))

    def get_action(self, state, timestep):
        """
        Find the cost of the sequences of actions and state that have shape [horizon, action dimension]
        and [horizon, state_dim], respectively.
        Use the function step of the environment: env.step(action). It returns, next_state, cost, done,
        env_infos.

        In order to set the environment at a specific state use the function self.env.set_state(state)
        """
        env = self.env
        horizon = min(self.H, env.H - timestep)

        def eval_mpc(actions, state):
            actions = actions.reshape(horizon, env.du)
            total_cost = 0
            """YOUR CODE HERE"""
            self.env.set_state(state)
            for i in range(horizon):
                action = actions[i]
                next_state, cost, done, env_infos = self.env.step(action)
                total_cost += cost

            """YOUR CODE ENDS HERE"""
            return total_cost

        self.init_actions = np.random.uniform(low=-.1, high=.1, size=(horizon * env.du,))
        res = minimize(lambda x: eval_mpc(x, state),
                       self.init_actions,
                       method='BFGS',
                       options={'xtol': 1e-6, 'disp': False, 'verbose': 2}
                       )

        print(res.message)
        print("The optimal cost is %.3f" % res.fun)
        act_shooting = res.x
        return act_shooting[:env.du]

    def reset(self):
        pass