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

from part_c import lqr_finite_horizon
from part_b import linearize_dynamics

# once again fill in the control input based on your controller
def simulate(K_lst, f, x_ref, u_ref, dt, n_starting_states, T, noise=None):
    def setup_heli_idx():
        idx = dict()
        k = 0
        keys = ["ned_dot", "ned", "pqr", "axis_angle"]
        for ky in range(len(keys)):
            idx[keys[ky]] = np.arange(k, k + 3)
            k += 3
        return idx

    idx = setup_heli_idx()

    def disp(sim, ref, label):
        cp = sns.color_palette("Paired")
        a, b = sim[idx[label]], ref[idx[label]]
        [plt.plot(a[i], linewidth=1, color=cp[i]) for i in range(a.shape[0])]
        [plt.plot(b[i], linewidth=2, linestyle=':', color=cp[i]) for i in range(b.shape[0])]
        legend_elements = [Line2D([0], [0], label='yours'), Line2D([0], [0], linestyle=':', label='target')]
        plt.legend(handles=legend_elements)
        plt.xlabel('time')
        plt.title(label)
        plt.show()

    for s in np.arange(n_starting_states):
        x, u = np.zeros((x_ref.shape[1], T)), np.zeros((u_ref.shape[1], T))
        x[:, 0] = starting_states[:, s]
        for t in np.arange(T - 1):
            """YOUR CODE HERE"""
            z_state_t = np.hstack([x[:,t] - x_ref[t, :], 1]) # Because of offset c
            u[:, t] = u_ref[t, :] + K_lst[T-2-t] @ (z_state_t)
            """YOUR CODE ENDS HERE"""
            x[:, t + 1] = f(x[:, t], u[:, t], dt)
            if noise is not None:
                x[:, t + 1] += noise[:, t]
        keys = ["ned_dot", "ned", "pqr", "axis_angle"]
        for key in keys:
            disp(x, x_ref.T, key)

if __name__ == "__main__":
    traj = loadmat("mats/heli_traj.mat")
    x_init, x_target, u_target = traj['x_init'], traj['x_target'], traj['u_target']

    plt.plot(x_target.T, linewidth = .6)
    plt.title("Visualization of Target Helicopter Trajectory")
    plt.xlabel("time")
    plt.show()

    f = sim_heli
    dt = 0.1  # we work with discrete time

    x_ref, u_ref = x_target.T, u_target.T
    my_eps = 0.001  # finite difference for numerical differentiation
    T, dx = x_ref.shape
    du = u_ref.shape[1]
    A_lst, B_lst = [], []  # this should look familiar, maybe your code from part (c) will be helpful!

    for t in range(T - 1):
        """YOUR CODE HERE"""
        A, B, c = linearize_dynamics(f, x_ref[t, :], u_ref[t,:], dt, my_eps, x_ref[t+1, :])
        A_t = A
        B_t = B
        """YOUR CODE ENDS HERE"""
        A_lst.append(A_t)
        B_lst.append(B_t)

    Q, R = np.eye(A_lst[0].shape[0]), np.eye(B_lst[0].shape[1])
    Q[-1, -1] = 0
    K_list, P_list = lqr_finite_horizon(A_lst, B_lst, Q, R, T - 1)  # you wrote this in part (c)


    # simulate (reference trajectory depicted by dotted lines)
    starting_states = x_init.reshape(-1, 1)
    simulate(K_list, f, x_ref, u_ref, dt, 1, T)

    # now with noise!
    simulate(K_list, f, x_ref, u_ref, dt, 1, T, noise=np.random.normal(scale=.1, size=(x_ref.shape[1], T)))