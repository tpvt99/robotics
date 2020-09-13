import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import math
from scipy.io import loadmat as loadmat


def kf_smooth(y, A, B, C, d, u, Q, R, init_x, init_V):
    '''
    function xfilt, xpred, Vfilt, loglik, xsmooth, Vsmooth, Q, R =
               kf_smooth(y, A, B, C, d, u, Q, R, init_x, init_V)


    Kalman filter
    xfilt, xpred, Vfilt, _, _, _, _, _ = kf_smooth(y_all, A, B, C, d, Q, R, init_x, init_V);

    Kalman filter with Smoother
    xfilt, xpred, Vfilt, loglik, xsmooth, Vsmooth, _, _ = kf_smooth(y_all, A, B, C, d, Q, R, init_x, init_V);

    Kalman filter with Smoother and EM algorithm
    xfilt, xpred, Vfilt, loglik, xsmooth, Vsmooth, Q, R = kf_smooth(y_all, A, B, C, d, Q, R, init_x, init_V);

    INPUTS:
    y - observations
    A, B, C, d:  x(:,t+1) = A x(:,t) + B u(:,t) + w(:,t)
                 y(:,t)   = C x(:,t) + d        + v(:,t)
    Q - covariance matrix of system x(t+1)=A*x(t)+w(t) , w(t)~N(0,Q)
    R - covariance matrix of output y(t)=C*x(t)+v(t) , v(t)~N(0,R)
    init_x - initial mean
    init_V - initial time


    OUTPUTS:
    xfilt = E[X_t|t]
    xpred - the filtered values at time t before measurement
    Vfilt - Cov[X_t|0:t]
    loglik - loglikelihood
    xsmooth - E[X_t|0:T]
    Vsmooth - Cov[X_t|0:T]
    Q - estimated system covariance according to 1 M step (of EM)
    R - estimated output covariance according to 1 M step (of EM)

    '''

    T = y.shape[1]
    ss = Q.shape[0]  # size of state space

    # Forward pass (Filter)
    # init the first values

    error_y = np.zeros([y.shape[0], 1, T])
    xpred = np.zeros([init_x.shape[0], init_x.shape[1], T])
    xfilt = np.zeros_like(xpred)
    Vpred = np.zeros([init_V.shape[0], init_V.shape[1], T])
    Vfilt = np.zeros_like(Vpred)


    for t in range(-1, T - 1):
        # dynamics update
        # P4(a) Filter
        if t == -1:  # handle the first step separately
            xpred[:, :, t + 1] = init_x
            Vpred[:, :, t + 1] = init_V
            loglik = 0
        else:
            '''Your code for P4(a) Kalman Filter '''
            # Hint: try something like u[:, t][..., np.newaxis] to fix shape issue
            xpred[:, :, t + 1] = A @ xfilt[:, :, t] + B @ u[:, t][..., np.newaxis]
            Vpred[:, :, t + 1] = A @ Vfilt[:, :, t] @ A.T + Q
            '''Your code end'''

        '''Your code for P4(a) Kalman Filter '''
        # Hint: you should follow the slides to compute xfilt and Vfilt
        error_y[:, :, t + 1] = y[:, t+1][..., np.newaxis] - (C @ xpred[:, :, t+1] + d)  # error (innovation)

        S =  C @ Vpred[:,:,t+1] @ C.T + R # Innovation (or residual) covariance: C Vpred_{t+1} C^T + R, you can ignore this temp var and write your own!
        K = Vpred[:,:,t+1] @ C.T @ np.linalg.pinv(S)  # Kalman gain matrix

        xfilt[:, :, t + 1] = xpred[:, :, t+1] + K @ (y[:, t+1][..., np.newaxis] - (C @ xpred[:, :, t+1] + d))
        Vfilt[:, :, t + 1] = Vpred[:, :, t+1] - K @ C @ Vpred[:, :, t+1]
        '''Your code end'''

        '''Your code for P4(b)(c) Kalman Smoother and EM '''
        # Hint: compute loglikelihood, note it is gaussian
        Sigma = C @ Vpred[:, :, t+1] @ C.T + R
        dd = error_y.shape[0]  # dimensions
        denom = (2 * math.pi) ** (np.linalg.matrix_rank(Sigma)/2) * np.sqrt(np.linalg.det(Sigma))
        # Hint: denom is used at the end of the next line. :)
        loglik = loglik + (
                    -1 / 2 * error_y[:, :, t + 1].T @ np.linalg.pinv(Sigma) @ error_y[:, :, t + 1] + np.log(1 / denom))
        '''Your code end'''

    # Backward pass (RTS Smoother and EM algorithm)
    # init the last values
    xsmooth = np.zeros_like(xfilt)
    Vsmooth = np.zeros_like(Vfilt)
    xsmooth[:, :, T - 1] = xfilt[:, :, T - 1]
    Vsmooth[:, :, T - 1] = Vfilt[:, :, T - 1]
    L = np.zeros_like(Vfilt)
    Q = Q * 0
    R = R * 0
    for t in range(T - 1, -1, -1):
        if t < T - 1:
            '''Your code for P4(b) Kalman Smoother '''
            # Hint: P4(b) Smoother
            L[:, :, t] = Vfilt[:, :, t] @ A.T @ np.linalg.pinv(Vpred[:, :, t+1]) # smoother gain matrix
            xsmooth[:, :, t] = xfilt[:, :, t] + L[:,:,t] @ (xsmooth[:, :, t+1] - xpred[:, :, t+1])
            Vsmooth[:, :, t] = Vfilt[:, :, t] + L[:,:,t] @ (Vsmooth[:, :, t+1] - Vpred[:, :, t+1]) @ L[:,:,t].T
            '''Your code end'''

            '''Your code for P4(c) the EM algorithm '''
            # P4(c) EM algorithm
            error_x = xsmooth[:, :, t+1] - A @ xsmooth[:, :, t] - B @ u[:, t][..., np.newaxis]
            P =  Vsmooth[:, :, t+1] - Vsmooth[:, :, t+1] @ L[:, :, t].T @ A.T - A @ L[:, :, t] @ Vsmooth[:, :, t+1]# some
            # temp var you can delete and write your own: Vsmooth[:, :, t+1] - Vsmooth[:, :, t+1] @ L[:, :, t].T @ A.T - A @ L[:, :, t] @ Vsmooth[:, :, t+1]
            Q = Q + error_x @ error_x.T + A @ Vsmooth[:, :, t] @ A.T + P
            e_y = y[:, t][..., np.newaxis] - C @ xsmooth[:, :, t] - d
            R = R + e_y @ e_y.T + C @ Vsmooth[:, :, t] @ C.T
            '''Your code end'''
    Q = Q / (T - 1)
    R = R / T

    return xfilt, xpred, Vfilt, loglik, xsmooth, Vsmooth, Q, R

def test_part4_a():
    """ When P4 (a) Kalman Filtering is done, please run this:  """
    for index in range(4):
        # data generation, whenever you want to run P4 (a)(b)(c), run this first!
        T, A, B, C, d, u, y, x = np.load(f'p3_a_data_{index + 1}.npy', allow_pickle=True)
        # now you should have variables:
        # T, A, B, C, d, u, y, x
        # They are described in the kf_smooth function. x is the groundtruth.

        x_init = np.zeros([5, 1]);  # mean at time t=1 before measurement at time t=1
        P_init = np.eye(5);  # covariance at time t=1 before measurement at time t=1

        # I found initially overestimating Q and R gives better learning of Q and R
        # during EM

        Q = 10 * np.eye(5);
        R = 10 * np.eye(2);
        ll = np.zeros(100)
        for i in range(100):
            xfilt, xpred, Vfilt, loglik, _, _, _, _ = kf_smooth(y, A, B, C, d, u, Q, R, x_init, P_init)

        plt.figure(figsize=(12, 6))
        for i in range(5):
            plt.plot(np.squeeze(x)[i, :], linewidth=1)
            plt.plot(np.squeeze(xfilt)[i, :], '-.', linewidth=1)
        plt.xlabel('timestep')
        plt.ylabel('state')
        plt.title('KF results')
        plt.show()

    # Please check the result in the plots. Plots for the first two datasets are provided as reference.

def test_part4_b():
    """ When P4 (b) Kalman Filtering is done, please run this:  """
    for index in range(4):
        # data generation, whenever you want to run P4 (a)(b)(c), run this first!
        T, A, B, C, d, u, y, x = np.load(f'p3_a_data_{index + 1}.npy', allow_pickle=True)
        # now you should have variables:
        # T, A, B, C, d, u, y, x
        # They are described in the kf_smooth function. x is the groundtruth.

        x_init = np.zeros([5, 1]);  # mean at time t=1 before measurement at time t=1
        P_init = np.eye(5);  # covariance at time t=1 before measurement at time t=1

        # I found initially overestimating Q and R gives better learning of Q and R
        # during EM

        Q = 10 * np.eye(5);
        R = 10 * np.eye(2);
        ll = np.zeros(100)

        for i in range(100):
            xfilt, xpred, Vfilt, loglik, xsmooth, Vsmooth, _, _ = kf_smooth(y, A, B, C, d, u, Q, R, x_init, P_init)
            ll[i] = loglik

        plt.figure(figsize=(12, 6))
        for i in range(5):
            plt.plot(np.squeeze(x)[i, :], linewidth=1)
            plt.plot(np.squeeze(xfilt)[i, :], '-.', linewidth=1)
            plt.plot(np.squeeze(xsmooth)[i, :], '--', linewidth=1)
        plt.xlabel('timestep')
        plt.ylabel('state')
        plt.title('KF results')
        plt.show()

    # Compare the Filtering and smoothing, which one is better? (No need to report).
    # Plots for the first two datasets are provided as reference.

def test_part4_c():
    """ When P4 (c) EM is done, please run this:  """
    for index in range(4):
        # data generation, whenever you want to run P4 (a)(b)(c), run this first!
        T, A, B, C, d, u, y, x = np.load(f'p3_a_data_{index + 1}.npy', allow_pickle=True)
        # now you should have variables:
        # T, A, B, C, d, u, y, x
        # They are described in the kf_smooth function. x is the groundtruth.

        x_init = np.zeros([5, 1]);  # mean at time t=1 before measurement at time t=1
        P_init = np.eye(5);  # covariance at time t=1 before measurement at time t=1

        # I found initially overestimating Q and R gives better learning of Q and R
        # during EM

        Q = 10 * np.eye(5);
        R = 10 * np.eye(2);
        ll = np.zeros(100)

        for i in range(100):
            xfilt, xpred, Vfilt, loglik, xsmooth, Vsmooth, Q, R = kf_smooth(y, A, B, C, d, u, Q, R, x_init, P_init)
            ll[i] = loglik

        if index > 1:
            break

        plt.figure(figsize=(12, 6))
        plt.plot(ll)
        plt.xlabel('iter')
        plt.ylabel('loglik')
        plt.title('Loglik')
        plt.show()

        plt.figure(figsize=(12, 6))
        for i in range(5):
            plt.plot(np.squeeze(x)[i, :], linewidth=1)
            plt.plot(np.squeeze(xfilt)[i, :], '-.', linewidth=1)
            plt.plot(np.squeeze(xsmooth)[i, :], '--', linewidth=1)
        plt.xlabel('timestep')
        plt.ylabel('state')
        plt.title('KF results')
        plt.show()

    # Hint: Note that loglik should be increasing.
    # Compare the filtering and smoothing results with the previous plots, do you see some difference? (No need to report)
    # Plots for the first two datasets are provided as reference.

if __name__ == '__main__':
    #test_part4_a()
    #test_part4_b()
    test_part4_c()
