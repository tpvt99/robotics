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




# implement the infinite horizon optimal feedback controller
def lqr_infinite_horizon(A, B, Q, R):
    """
    find the infinite horizon K and P through running LQR back-ups
    until l2-norm(K_new - K_curr, 2) <= 1e-4
    return: K, P
    """

    dx, du = A.shape[0], B.shape[1]
    P, K_current = np.eye(dx), np.zeros((du, dx))

    """YOUR CODE HERE"""
    while True:
        K_new = -np.matmul(np.linalg.inv(R + np.matmul(np.matmul(B.T, P), B)), np.matmul(np.matmul(B.T, P), A))
        P = Q + np.matmul(np.matmul(K_new.T, R), K_new) + \
            np.matmul(np.matmul(np.transpose(A + np.matmul(B, K_new)), P), A + np.matmul(B, K_new))
        if np.linalg.norm(K_new - K_current, ord=2) <= 1e-4:
            break

        K_current = K_new

    """YOUR CODE ENDS HERE"""
    return K_new, P




# fill in the simulation to use your controller, K_inf, at each timestep then run the cell to generate plots
def simulate(A, B, K_inf, n_starting_states, T, noise=None):
    for s in np.arange(n_starting_states):
        x, u = np.zeros((K_inf.shape[1], T + 1)), np.zeros((K_inf.shape[0], T + 1))
        x[:, 0] = starting_states[:, s]
        for t in np.arange(T):
            """YOUR CODE HERE"""
            u[:, t] = K_inf @ x[:, t]
            """YOUR CODE ENDS HERE"""
            x[:, t + 1] = A @ x[:, t] + B @ u[:, t]
            if noise is not None:
                x[:, t + 1] += noise[:, t]
        plt.plot(x.T, linewidth=.7)
        plt.xlabel('time')
        plt.title("Noisy Linear System Start State #{}".format(s)) if noise is not None else plt.title(
            "Linear System Start State #{}".format(s))
        plt.legend(["dim" + str(i) for i in range(len(x))])
        plt.show()


if __name__ == "__main__":
    A = np.array([[0.0481, -0.5049, 0.0299, 2.6544, 1.0608],
                  [2.3846, -0.2312, -0.1260, -0.7945, 0.5279],
                  [1.4019, -0.6394, -0.1401, 0.5484, 0.1624],
                  [-0.0254, 0.4595, -0.0862, 2.1750, 1.1012],
                  [0.5172, 0.5060, 1.6579, -0.9407, -1.4441]])
    B = np.array([[-0.7789, -1.2076],
                  [0.4299, -1.6041],
                  [0.2006, -1.7395],
                  [0.8302, 0.2295],
                  [-1.8465, 1.2780]])
    dx = A.shape[0]
    du = B.shape[1]

    # verify the above statement
    lst = [B]
    """YOUR CODE HERE"""
    AB = np.matmul(A, B)
    A2B = np.matmul(A, AB)
    A3B = np.matmul(A, A2B)
    A4B = np.matmul(A, A3B)
    lst = [B, AB, A2B, A3B, A4B]

    """YOUR CODE ENDS HERE"""
    np.linalg.matrix_rank(np.hstack(lst))

    # problem has been defined, let's solve it:
    Q, R = np.eye(dx), np.eye(du)
    K_inf, P_inf = lqr_infinite_horizon(A, B, Q, R)

    starting_states = np.array([[-1.9613, 1.9277, -0.2442],
                                [-1.3127, -0.2406, -0.0260],
                                [0.0698, -0.5860, -0.7522],
                                [0.0935, -0.1524, -0.9680],
                                [1.2494, 0.5397, -0.5146]])
    n_starting_states = starting_states.shape[1]
    T = 20  # simulating for 20 steps
    simulate(A, B, K_inf, n_starting_states, T)

    # and in the presence of noise:
    noise_id = "p_a_w"
    T = 100  # simulating for 100 steps
    simulate(A, B, K_inf, n_starting_states, T, noise=loadmat("mats/" + noise_id + ".mat")[noise_id])