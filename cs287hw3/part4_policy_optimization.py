import numpy as np
from envs.cart_pole_env import CartPoleEnv
from utils import *

def eval_policy(env, policy, params):
    """
    Find the cost the policy with parameters params.
    Use the function step of the environment: env.step(action). It returns: next_state, cost, done,
    env_infos.

    You can set the parameters of the policy by policy.set_params(params) and get the action for the current state
    with policy.get_action(state).
    """
    state = env.reset()
    total_cost = 0
    horizon = env.H

    """YOUR CODE HERE"""
    policy.set_params(params)
    for i in range(horizon):
        action = policy.get_action(state)
        next_state, cost, done, env_infos = env.step(action)
        total_cost += cost
        state = next_state

    """YOUR CODE ENDS HERE"""
    return total_cost

def minimize_policy_shooting(env):
    policy_shooting = NNPolicy(env.dx, env.du, hidden_sizes=(10, 10))
    policy_shooting.init_params()
    params = policy_shooting.get_params()
    res = minimize(lambda x: eval_policy(env, policy_shooting, x),
                   params,
                   method='BFGS',
                   options={'xtol': 1e-6, 'disp': False, 'verbose': 2})
    print(res.message)
    print("The optimal cost is %.3f" % res.fun)
    params_shooting = res.x
    policy_shooting.set_params(params_shooting)
    return policy_shooting

if __name__ == "__main__":
    env = CartPoleEnv()
    policy_shooting = minimize_policy_shooting(env)