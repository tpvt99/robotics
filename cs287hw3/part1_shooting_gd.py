import numpy as np
import scipy
from utils import *
from linear_environment import LinearEnv


def eval_shooting(actions, env):
    """
    Find the cumulative cost of the sequences of actions, which has shape [horizon, action dimension].
    Use the function step of the environment: env.step(action). It returns: next_state, cost, done,
    env_infos.
    """

    state = env.reset()
    actions = actions.reshape(env.H, env.du)
    horizon = env.H

    total_cost = 0
    env.set_state(state)

    """YOUR CODE HERE"""
    for i in range(horizon):
        action = actions[i]
        state, cost, done, _ = env.step(action)
        total_cost += cost

    """YOUR CODE ENDS HERE"""
    return total_cost

# Gradient Descent: u = u - step * (R * u)
def minimize_shooting(env, init_actions=None):
    if init_actions is None:
        init_actions = np.random.uniform(low=-.1, high=.1, size=(env.H * env.du,))
    """YOUR CODE HERE"""

    step_size = 1e-2
    epoch = 10000
    actions = init_actions
    for i in range(epoch):
        costs = eval_shooting(actions, env)
        actions = actions.reshape(env.H, env.du)
        actions = actions - step_size * np.matmul(actions, env.R)
        actions = actions.reshape(-1)
        if i % 10 == 0:
            print("Iter {} with cost {}".format(i, costs))

    act_shooting = actions
    policy_shooting = ActPolicy(env=env,
                                actions=act_shooting
                               )
    return policy_shooting
"""YOUR CODE ENDS HERE"""


if __name__ == "__main__":
    env = LinearEnv(multiplier=10.)
    policy_shooting = minimize_shooting(env)