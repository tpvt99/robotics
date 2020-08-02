import numpy as np
import scipy
from utils import *
from linear_environment import LinearEnv
from part1_shooting import minimize_shooting
import matplotlib.pyplot as plt

def eval_collocation(env, x):
    """
    Find the cost of the sequences of actions and state that have shape [horizon, action dimension]
    and [horizon, state_dim], respectively.
    Use the function step of the environment: env.step(action). It returns: next_state, cost, done,
    env_infos.
    In order to set the environment at a specific state use the function env.set_state(state)
    """
    state = env.reset()
    total_cost = 0
    states, actions = x[:env.H * env.dx], x[env.H * env.dx:]
    states = states.reshape(env.H, env.dx)
    actions = actions.reshape(env.H, env.du)
    horizon = env.H

    """YOUR CODE HERE"""
    for i in range(horizon):
        action = actions[i]
        if i > 0:
            env.set_state(states[i-1])
        next_state, cost, done, _ = env.step(action)
        total_cost += cost

    """YOUR CODE ENDS HERE"""
    return total_cost


def constraints(env, x):
    """
    In optimization, the equality constraints are usually specified as h(x) = 0. In this case, we would have
    x_{t+1} - f(x_t, u_t) = 0. Here, you have to create a list that contains the value of the different
    constraints, i.e., [x_0 - f(x_init, u_0), x_1 - f(x_0, u_1),..., x_H - f(x_{H-1}, u_{H-1})].
    Use the function env.set_state(state) to set the state to the variable x_t.
    Use the function step of the environment: env.step(action), which returns next_state, cost, done,
    env_infos; to obtain x_{t+1}.
    """
    state = env.reset()
    constraints = []
    states, actions = x[:env.H * env.dx], x[env.H * env.dx:]
    states = states.reshape(env.H, env.dx)
    actions = actions.reshape(env.H, env.du)
    horizon = env.H

    """YOUR CODE HERE"""
    for i in range(horizon):
        action = actions[i]
        state = states[i]
        if i > 0: # from the 2nd iters, we set state to x_0 and find f(x_0, u_1). 1st iter has the x_init as the state
            env.set_state(states[i - 1])
        next_state, cost, done, _ = env.step(action)
        constraints.append(next_state - state)

    """YOUR CODE ENDS HERE"""
    return np.concatenate(constraints)

def minimize_collocation(env, init_states_and_actions=None):
    if init_states_and_actions is None:
        init_states_and_actions = np.random.uniform(low=-.1, high=.1, size=(env.H * (env.du + env.dx),))

    """YOUR CODE HERE"""
    eq_cons = {'type': 'eq',
               'fun' : lambda x: constraints(env, x)
              }

    res = minimize(fun= lambda x : eval_collocation(env, x),
                   x0= init_states_and_actions,
                   method='SLSQP',
                   constraints=eq_cons,
                   options={'xtol': 1e-6, 'disp': False, 'verbose': 0, 'maxiter':201}
                  )
    print(res.message)
    print("The optimal cost is %.3f" % res.fun)
    states_collocation, act_collocation = res.x[:env.H * env.dx], res.x[env.H * env.dx:]
    states_collocation = states_collocation.reshape(env.H, env.dx)
    policy_collocation = ActPolicy(env,
                                   actions=act_collocation)
    """YOUR CODE ENDS HERE"""
    return policy_collocation, states_collocation

if __name__ == "__main__":

    env = LinearEnv()
    policy_collocation, states_collocation = minimize_collocation(env)
    policy_shooting = minimize_shooting(env)

    cost_shoot, states_shoot = rollout(env, policy_shooting)
    cost_col, states_col = rollout(env, policy_collocation)
    states_shoot, states_col = np.array(states_shoot), np.array(states_col)
    error = np.linalg.norm(states_col - np.array(states_collocation))
    ts = np.arange(states_shoot.shape[0])
    print("---- Quantitative Metrics ---")
    print("Shooting Cost %.3f" % cost_shoot)
    print("Collocation Cost %.3f" % cost_col)
    print("Collocation Error %.3f" % error)

    print("\n\n---- Qualitative Metrics ---")
    print("Evolution of the value of each dimension across 20 timesteps for the shooting methods.")
    print("Both methods converge to the origin. Shooting: solid line(-);  Collocation: dashed line(--).")

    for i in range(env.dx):
        plt.plot(ts, states_shoot[:, i], '-', ts, states_col[:, i], '--')
    plt.show()

