import numpy as np
import matplotlib.pyplot as plt

# Numerical Jacobian of func
# idx specifies the index of argument w.r.t which the Jacobian is computed
# the rest are arguments passed to func

# For instance, for y = f(x1, x2, ..., xN)
# numerical_jac(@f, 2, x1, x2, ..., xN) computes the Jacobian df/dx2

def numerical_jac(func, idx, var_list):
    step = 1e-6

    x = var_list[idx]
    y = func(*var_list)
    lenx = len(x)
    leny = len(y)
    J = np.zeros([leny, lenx])

    for i in range(lenx):
        xhi = x[i] + step
        xlo = x[i] - step

        var_list[idx][i] = xhi
        yhi = func(*var_list)
        var_list[idx][i] = xlo
        ylo = func(*var_list)
        var_list[idx][i] = x[i]
        J[:,i] = np.squeeze((yhi - ylo)/(xhi - xlo))

    return J

def ekf(x_t, Sigma_t, u_t, z_tp1, model):
    xDim = model.xDim
    qDim = model.qDim
    rDim = model.rDim
    Q = model.Q
    R = model.R
    '''Your code here:'''
    A = numerical_jac(model.dynamics_func, 0, [x_t, u_t, np.zeros([qDim,1])])
    M = numerical_jac(model.dynamics_func, 2, [x_t, u_t, np.zeros([qDim,1])])
    Sigma_tp1 = A @ Sigma_t @ A.T + M @ Q @ M.T# write your Sigma t plut 1
    x_tp1 = model.dynamics_func(x_t, u_t, np.zeros([qDim,1])) # forward with dynamcis

    H = numerical_jac(model.obs_func, 0, [x_tp1, np.zeros([rDim,1])])
    N = numerical_jac(model.obs_func, 1, [x_tp1, np.zeros([rDim,1])])

    K = Sigma_tp1 @ H.T @ np.linalg.inv(H @ Sigma_tp1 @ H.T + N @ R @ N.T)

    x_tp1 = x_tp1 + K @ (z_tp1 - model.obs_func(x_tp1, np.zeros([rDim,1])))
    Sigma_tp1 = (np.eye(xDim) - K @ H) @ Sigma_tp1
    '''Your code end'''
    return x_tp1, Sigma_tp1


def plot_1d_trajectory(mean_ekf, cov_ekf, X, model):
    hor = np.array(range(model.T))

    # Iterate over dimensions
    for d in range(model.xDim):
        plt.figure(figsize=(12, 6))
        x_td = np.squeeze(mean_ekf[d, :, :]).T
        Sigma_td = np.squeeze(cov_ekf[d, d, :])

        ff1 = np.hstack([hor, np.flip(hor)])
        ff2 = np.hstack([(x_td + 3 * np.sqrt(Sigma_td)),
                         np.flip((x_td - 3 * np.sqrt(Sigma_td)))])
        plt.fill_between(ff1, ff2, alpha=0.5)
        plt.plot(hor, X[d, :], 'rs-', linewidth=3)  # ground truth
        plt.plot(hor, x_td, 'b*-.', linewidth=1)
        plt.plot(hor, x_td + 3 * np.sqrt(Sigma_td), color=np.array([0, 4, 0]) / 8)
        plt.plot(hor, x_td - 3 * np.sqrt(Sigma_td), color=np.array([0, 4, 0]) / 8)
        plt.xlabel('time steps')
        plt.ylabel('state')
        plt.show()


# test ekf

# Setup model
class Model():
    def __init__(self):
        # Setup model dimensions
        self.xDim = 2  # state space dimension
        self.uDim = 2  # control input dimension
        self.qDim = 2  # dynamics noise dimension
        self.zDim = 2  # observation dimension
        self.rDim = 2  # observation noise dimension
        self.Q = 2 * np.eye(self.qDim)  # dynamics noise variance
        self.R = np.eye(self.rDim)  # observation noise variance
        self.R[1, 1] = 10
        self.T = 50  # number of time steps in trajectory

    # Dynamics function: x_t+1 = dynamics_func(x_t, u_t, q_t, model)
    def dynamics_func(self, x_t, u_t, q_t):
        x_tp1 = np.zeros([self.xDim, 1])
        x_tp1[0] = 0.1 * (x_t[0] * x_t[0]) - 2 * x_t[0] + 20 + q_t[0]
        x_tp1[1] = x_t[0] + 0.3 * x_t[1] - 3 + q_t[1] * 3
        return x_tp1

    # Observation function: z_t = obs_func(x_t, r_t, model)
    def obs_func(self, x_t, r_t):
        z_t = np.zeros([self.zDim, 1])
        z_t[0] = (x_t.T @ x_t) + np.sin(5 * r_t[0])
        z_t[1] = 3 * (x_t[1] * x_t[1]) / x_t[0] + r_t[1]
        return z_t

    def load_states_observations(self, i):
        X, Z = np.load(f'p6_data_{i}.npy', allow_pickle=True)
        return X, Z


model = Model()

x0 = np.array([[10], [10]])
Sigma0 = np.eye(model.xDim)
for index in range(4):
    X, Z = model.load_states_observations(index)

    # Mean and covariances for plotting
    mean_ekf = np.zeros([model.xDim, 1, model.T])
    cov_ekf = np.zeros([model.xDim, model.xDim, model.T])

    mean_ekf[:, :, 0] = x0
    cov_ekf[:, :, 0] = Sigma0

    for t in range(model.T - 1):
        mean_ekf[:, :, t + 1], cov_ekf[:, :, t + 1] = ekf(mean_ekf[:, :, t], cov_ekf[:, :, t],
                                                          np.zeros([model.uDim, 1]), Z[:, t + 1][..., np.newaxis],
                                                          model)

    plot_1d_trajectory(mean_ekf, cov_ekf, X, model)

    print(f'Mean at last timestep: {mean_ekf[:, :, model.T - 1]}')
    print(f'Covariance matrix at last timestep: {cov_ekf[:, :, model.T - 1]}')

# The plot for the initial two data-sets are provided as reference.
