from utils import *
import matplotlib.pyplot as plt
from envs.cart_pole_env import CartPoleEnv
from part1_shooting import minimize_shooting
from part4_model_predictive_control import MPCPolicy
from part4_policy_optimization import minimize_policy_shooting


env = CartPoleEnv()
print("Initializing action shooting: ")
action_shooting = minimize_shooting(env)
print("Initializing MPC Policy")
mpc_policy = MPCPolicy(env, env.H)
print("Initializing Policy Shooting: ")
policy_shooting = minimize_policy_shooting(env)


noise = 1.
cost_act, states_act = rollout(env, action_shooting, noise)
cost_pi, states_pi = rollout(env, policy_shooting, noise)
cost_mpc, states_mpc = rollout(env, mpc_policy, noise)
states_act, states_pi, states_mpc = np.array(states_act), np.array(states_pi), np.array(states_mpc)
print("---- Quantitative Metrics ---")
print("Action Cost %.3f" % cost_act)
print("Policy Cost %.3f" % cost_pi)
print("MPC Cost %.3f" % cost_mpc)

print("\n\n---- Qualitative Metrics ---")
print("Evolution of the value of the angle and angular velocity of the cart-pole environment across 50 timesteps for the open-loop, policy controller, and mpc controller.")
print("All the approaches achieve the same cost and follow the same trajectory. Open-loop: solid line(-);  Policy: dashed line(--). MPC: dotted line(.)")
ts = np.arange(states_act.shape[0])
plt.plot(ts, states_act[:, 2], '-', ts, states_pi[:, 2], '--', states_mpc[:, 2], '.')
plt.plot(ts, states_act[:, 3], '-', ts, states_pi[:, 3], '--', states_mpc[:, 3], '.')
plt.show()