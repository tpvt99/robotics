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
        constraints.append(state - next_state)

    """YOUR CODE ENDS HERE"""
    return np.concatenate(constraints)


# Cost = 1/2 * sum (xQx + uRu) + mu * sum_t(xt+1 - A * x_t - B * u_t)
# dCost/du = R*u -B * (x_t+1 - A * x_t - B * u_t)/abs(xt+1 - A * x_t - B * u_t)
# dCost/dx = Q*x +  (x_t - A * x_t-1 - B * u_t-1)/abs(x_t - A * x_t-1 - B * u_t-1)
# - A * (x_t+1 - A * x_t - B * u_t)/abs(x_t+1 - A * x_t - B * u_t)

def minimize_collocation(env, init_states_and_actions=None):
    if init_states_and_actions is None:
        init_states_and_actions = np.random.uniform(low=-.1, high=.1, size=(env.H * (env.du + env.dx),))

    mu = 1
    mu_t = 1.5
    epsilon = 1e-3
    step_size = 1e-5
    iter = 0
    states_and_actions = init_states_and_actions

    while np.sum(np.abs(constraints(env, states_and_actions))) >= epsilon:
        mu = mu_t * mu

        while True:
            states, actions = states_and_actions[:env.H * env.dx], states_and_actions[env.H * env.dx:]
            old = states_and_actions
            costs = eval_collocation(env, states_and_actions)
            actions = actions.reshape(env.H, env.du)
            states = states.reshape(env.H, env.dx)

            v = constraints(env, states_and_actions)
            v = v/np.abs(v)
            v_roll = np.roll(v, -1 * env.dx)
            v_roll[-1 * env.dx:] = 0
            v = v.reshape(env.H, env.dx)
            v_roll = v_roll.reshape(env.H, env.dx)

            states_gradients = np.matmul(states, env.Q) + mu * (v + np.matmul(v_roll, -env.A))
            actions_gradients = np.matmul(actions, env.R) + mu * np.matmul(v, -env.B)
            actions = actions - step_size * actions_gradients
            states = states - step_size * states_gradients
            actions = actions.reshape(-1)
            states = states.reshape(-1)
            states_and_actions = np.concatenate([states, actions])
            print("Iter {} with cost {} and constraints {}".format(iter, costs,
                np.sum(np.abs(constraints(env, states_and_actions)))))
            iter+=1

            if np.abs(np.sum(np.abs(constraints(env, states_and_actions))) - \
                      np.sum(np.abs(constraints(env, old)))) <= 1e-3:
                print('Small then break')
                break


    states_collocation, act_collocation = states_and_actions[:env.H * env.dx], states_and_actions[env.H * env.dx:]
    states_collocation = states_collocation.reshape(env.H, env.dx)
    policy_collocation = ActPolicy(env,
                                   actions=act_collocation)
    """YOUR CODE ENDS HERE"""
    return policy_collocation, states_collocation

if __name__ == "__main__":

    env = LinearEnv(multiplier=1)
    policy_collocation, states_collocation = minimize_collocation(env)

