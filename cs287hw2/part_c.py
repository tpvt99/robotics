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


# implement a finite horizon optimal feedback controller, accounting for possibly time-varying parameters
def lqr_finite_horizon(A_lst, B_lst, Q_lst, R_lst, T):
    """
    Each of A_lst, B_lst, Q_lst, and R_lst is either a python list (of length T) of numpy arrays
        or a numpy array (indicating this parameter is not time-varying).
    You will need to handle both cases in your implementation

    Find the finite horizon K and P through running LQR back-ups
    return: K_{1:T}, P_{1:T}
    """

    K_lst, P_lst = [], []
    """YOUR CODE HERE"""
    # check if at least one is time-varying
    if isinstance(A_lst, list) or isinstance(B_lst, list) or isinstance(Q_lst, list) or isinstance(R_lst, list):
        T = T
    else:
        T = None

    dx = A_lst[0].shape[0] if isinstance(A_lst, list) else A_lst.shape[0]
    P = np.eye(dx)

    if T is not None:
        for t in range(T):
            A_prev = A_lst[T-1-t] if isinstance(A_lst, list) else A_lst
            B_prev = B_lst[T-1-t] if isinstance(B_lst, list) else B_lst
            Q_prev = Q_lst[T-1-t] if isinstance(Q_lst, list) else Q_lst
            R_prev = R_lst[T-1-t] if isinstance(R_lst, list) else R_lst

            K_t = -np.matmul(np.linalg.inv(R_prev + np.matmul(np.matmul(B_prev.T, P), B_prev)),
                               np.matmul(np.matmul(B_prev.T, P), A_prev))
            P_t = Q_prev + np.matmul(np.matmul(K_t.T, R_prev), K_t) + \
                np.matmul(np.matmul(np.transpose(A_prev + np.matmul(B_prev, K_t)), P), A_prev + np.matmul(B_prev, K_t))
            P = P_t
            K_lst.append(K_t)
            P_lst.append(P_t)

    else:
        K_lst, P_lst = lqr_infinite_horizon(A_lst, B_lst, Q_lst, R_lst)

    """YOUR CODE ENDS HERE"""
    return K_lst, P_lst

# fill in to use your controller
def simulate(A_lst, B_lst, K_list, n_starting_states, T, noise=None):
    for s in np.arange(n_starting_states):
        x, u = np.zeros((K_list[0].shape[1], T + 1)), np.zeros((K_list[0].shape[0], T + 1))
        x[:, 0] = starting_states[:, s]
        for t in np.arange(T):
            """YOUR CODE HERE"""
            u[:, t] = K_lst[T-1-t] @ x[:, t]
            """YOUR CODE ENDS HERE"""
            x[:, t + 1] = A_lst[t] @ x[:, t] + B_lst[t] @ u[:, t]
            if noise is not None:
                x[:, t + 1] += noise[:, t]

        plt.plot(x.T, linewidth=.7)
        plt.plot(np.squeeze(u.T), linewidth=.7, linestyle='--')
        legend_elements = [Line2D([0], [0], label='x'), Line2D([0], [0], linestyle='--', label='u')]
        plt.legend(handles=legend_elements)
        plt.xlabel('time')
        plt.title("LTV Sanity Check")
        plt.show()

if __name__ == "__main__":
    # here we define a LTV system for a fixed horizon
    T = 100
    A_lst = [np.array([[np.sin(t), -0.5049, 0.0299, 2.6544, 1.0608],
                       [2.3846, -0.2312, -0.1260, -0.7945, 0.5279],
                       [1.4019, -0.6394, -0.1401, 0.5484, 0.1624],
                       [-0.0254, 0.4595, -0.0862, 2.1750, 1.1012],
                       [0.5172, 0.5060, 1.6579, -0.9407, -1.4441]]) for t in range(T)]
    B_lst = [np.array([[-0.7789, -1.2076],
                       [0.4299, -1.6041],
                       [0.2006, -1.7395],
                       [0.8302, 0.2295],
                       [-1.8465, np.cos(t)]]) for t in range(T)]

    starting_states = np.array([[-1.9613, 1.9277, -0.2442],
                                [-1.3127, -0.2406, -0.0260],
                                [0.0698, -0.5860, -0.7522],
                                [0.0935, -0.1524, -0.9680],
                                [1.2494, 0.5397, -0.5146]])
    n_starting_states = starting_states.shape[1]

    dx, du = A_lst[0].shape[0], B_lst[0].shape[1]
    Q, R = np.eye(dx), np.eye(du)
    K_lst, P_lst = lqr_finite_horizon(A_lst, B_lst, Q, R, T)

    # simulate to sanity check your TV solution
    simulate(A_lst, B_lst, K_lst, n_starting_states, T)