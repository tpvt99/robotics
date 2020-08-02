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

def minimize_shooting(env, init_actions=None):
    if init_actions is None:
        init_actions = np.random.uniform(low=-.1, high=.1, size=(env.H * env.du,))
    """YOUR CODE HERE"""
    res = minimize(fun= eval_shooting,
               x0= init_actions,
               args = (env),
               method='BFGS',
               options={'xtol': 1e-6, 'disp': False, 'verbose': 2}
              )

    act_shooting = res.x
    print(res.message)
    print("The optimal cost is %.3f" % res.fun)
    policy_shooting = ActPolicy(env=env,
                                actions=act_shooting
                               )
    return policy_shooting
"""YOUR CODE ENDS HERE"""


if __name__ == "__main__":
    env = LinearEnv()
    policy_shooting = minimize_shooting(env)