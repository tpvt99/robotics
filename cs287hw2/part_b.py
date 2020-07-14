from IPython import display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from gym.envs.mujoco import *
from envs.hopper_env import HopperModEnv
from envs.cheetah_env import CheetahModEnv
import numpy as np
import copy
import gym
from scipy.io import loadmat
from scipy.io import savemat
import moviepy.editor as mpy
from simulators import *
from rot_utils import *
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
from part_a import lqr_infinite_horizon

# implement linearization about a point
def linearize_dynamics(f, x_ref, u_ref, dt, my_eps, x_ref_tplus1=None):
    """
    f : dynamics simulator
    my_eps : delta for forward and backward differences you'll need
    NOTE: please use centered finite differences!

    x(:,t+1) - x_ref  approximately = A*( x(:,t)-x_ref ) + B* ( u(:,t) - u_ref ) + c
    If we pick x_ref and u_ref to constitute a fixed point, then c == 0

    For part (b), you do not need to use the optional argument (nor c).
    For part (d), you'll have to revisit and modify this function
        --at this point, you'll want to use the optional argument and the resulting c.

    return: A, B, c
    """

    if x_ref_tplus1 is not None:
        x_ref_next = x_ref_tplus1
    else:
        x_ref_next = x_ref

    dx, du = x_ref.shape[0], u_ref.shape[0]
    A, B = np.zeros((dx, dx)), np.zeros((dx, du))

    """YOUR CODE HERE"""
    for i in range(dx):
        delta_x = np.zeros((dx,))
        delta_x[i] = my_eps
        f_x_forward = f(x_ref + delta_x, u_ref, dt)
        f_x_backward = f(x_ref - delta_x, u_ref, dt)
        f_derivatives = (f_x_forward - f_x_backward) / (2 * my_eps)
        A[:, i] = f_derivatives

    for j in range(du):
        delta_u = np.zeros((du,))
        delta_u[j] = my_eps
        f_x_forward = f(x_ref, u_ref + delta_u, dt)
        f_x_backward = f(x_ref, u_ref - delta_u, dt)
        f_derivatives = (f_x_forward - f_x_backward) / (2 * my_eps)
        B[:, j] = f_derivatives

    """YOUR CODE ENDS HERE"""

    c = f(x_ref, u_ref, dt) - x_ref_next
    if x_ref_tplus1 is not None:
        A = np.hstack([A, c.reshape(-1,1)])
        A = np.vstack([A, np.zeros(A.shape[1])])
        A[-1,-1] = 1
        B = np.vstack([B, np.zeros(B.shape[1])])

    if len(B.shape) == 1:
        return A, B.reshape(-1, 1), c
    return A, B, c


# take an environment and find the infinite horizon controller for the linearized system
def lqr_nonlinear(config):
    env = config['env']
    f = config['f']
    dt = 0.1  # we work with discrete time
    my_eps = 0.01  # finite difference for numerical differentiation

    # load in our reference points
    x_ref, u_ref = config['x_ref'], config['u_ref']

    # linearize
    A, B, c = linearize_dynamics(f, x_ref, u_ref, dt, my_eps)
    print('A shape: ', A.shape, 'B shape: ', B.shape)
    dx, du = A.shape[0], B.shape[1]
    Q, R = np.eye(dx), np.eye(du) * 2

    # solve for the linearized system
    K_inf, P_inf = lqr_infinite_horizon(A, B, Q, R)  # you implemented in part (a)

    # recognize the simulation code from part (a)? modify it to use your controller at each timestep
    def simulate(K_inf, f, x_ref, u_ref, dt, n_starting_states, T, noise=None):
        for s in np.arange(n_starting_states):
            x, u = np.zeros((K_inf.shape[1], T + 1)), np.zeros((K_inf.shape[0], T + 1))
            x[:, 0] = starting_states[:, s]
            for t in np.arange(T):
                """YOUR CODE HERE"""
                u[:, t] = u_ref + K_inf @ (x[:, t] - x_ref)
                """YOUR CODE ENDS HERE"""
                x[:, t + 1] = f(x[:, t], u[:, t], dt)
                if "p_val" in config.keys():
                    perturbation_values = config["p_val"]
                    perturb = perturbation_values[t // (T // len(perturbation_values))]
                    x[:, t + 1] = f(x[:, t], u[:, t], dt, rollout=True, perturb=perturb)
                if env is not None:
                    if t % 5 == 0:
                        plt.clf()
                        plt.axis('off')
                        plt.grid(b=None)
                        plt.imshow(env.render(mode='rgb_array', width=256, height=256))
                        plt.title("Perturbation Magnitude {}".format(perturb))
                        display.clear_output(wait=True)
                        display.display(plt.gcf())

                if noise is not None:
                    x[:, t + 1] += noise[:, t]
            if env is not None:
                plt.clf()

            plt.plot(x.T[:-1], linewidth=.6)
            plt.plot(np.squeeze(u.T[:-1]) / 10.0, linewidth=.7, linestyle='--')  # scaling for clarity
            if 'legend' in config.keys():
                config['legend'].append('u')
                plt.legend(config['legend'])
            else:
                legend_elements = [Line2D([0], [0], label='x'), Line2D([0], [0], linestyle='--', label='u')]
                plt.legend(handles=legend_elements)
            plt.xlabel('time')
            plt.title(config["exp_name"])
            plt.show()

    # now let's simulate and see what happens for a few different starting states
    starting_states = config['ss']
    n_starting_states = starting_states.shape[1]
    T = config['steps']  # simulating for T steps
    simulate(K_inf, f, x_ref, u_ref, dt, n_starting_states, T)
    if 'noise' in config.keys():
        # and now in the presence of noise
        noise_id = config['noise']
        noise_loaded = loadmat("mats/" + noise_id + ".mat")[noise_id]
        simulate(K_inf, f, x_ref, u_ref, dt, n_starting_states, noise_loaded.shape[1], noise=noise_loaded)

if __name__ == "__main__":

    runCartPole = 1
    runHelicopter = 0
    runHopper = 0
    if runCartPole:
        # Find the infinite horizon controller for the linearized version of the cartpole balancing problem
        cartpole_config = {
            'f': sim_cartpole,
            'exp_name': "Cartpole-Balancing",
            'env': None,
            'steps': 500,
            'x_ref': np.array([0, np.pi, 0, 0]),
            'u_ref': np.array([0]),
            'legend':['x', 'theta', 'xdot', 'thetadot'],
            'ss': np.array([[0, 0, 0, 10, 50],
                            [9*np.pi/10, 3*np.pi/4, np.pi/2, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]), #ss = starting states
            'noise': 'p_b_w',
        }
        lqr_nonlinear(cartpole_config)

    if runHelicopter:
        # Find the infinite horizon controller for the linearized version of the hovering copter
        # Just run the cell below to generate plots using the code you wrote for cartpole!
        x_ref, u_ref = np.zeros(12), np.zeros(4)
        x_ref[9] = np.arcsin(3.0 / (5 * 9.81))
        u_ref[3] = 9.81 * 5 * np.cos(x_ref[9]) / 137.5
        heli_config = {
            'f': sim_heli,
            'env': None,
            'exp_name': "Helicopter-Hovering",
            'steps': 200,
            'x_ref': x_ref,
            'u_ref': u_ref,
            'ss': loadmat("mats/p_c_heli_starting_states.mat")["heli_starting_states"],  # ss = starting states
            'noise': 'p_c_w',
        }
        lqr_nonlinear(heli_config)

    if runHopper:
        env = HopperModEnv()
        x_ref, u_ref = np.zeros(11), np.zeros(env.action_space.sample().shape[0])
        hopper_config = {
            'env': env,
            'f': env.f_sim,
            'exp_name': "Perturbed Hopper",
            'steps': 500,
            'x_ref': x_ref,
            'u_ref': u_ref,
            'ss': np.array([[np.concatenate([env.init_qpos[1:], env.init_qvel])]]),
            'p_val': [0, .1, 1, 10]
        }
        lqr_nonlinear(hopper_config)