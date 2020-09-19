import numpy as np
import matplotlib.pyplot as plt
from part4 import kf_smooth

def time_and_meas_update(Sigma, A, Sigma_w, C, R):
    '''Your code here'''
    # all the definitions are similar to KF implementation, performing time update and measurement update
    # here Sigma_w is Q in the slides. You need to output the Sigma for the measurement update.
    n = A.shape[0]
    Sigma_new = Sigma # this line is given
    Sigma_new = A @ Sigma_new @ A.T + Sigma_w
    S = C @ Sigma_new @ C.T + R
    K = Sigma_new @ C.T @ np.linalg.inv(S)
    Sigma_new = (np.identity(n) - K @ C) @ Sigma_new
    '''your code end'''
    return Sigma_new


# initialize the three sensors
n=3
A  = np.array([[-0.6, 0.8, 0.5], [-0.1, 1.5, -1.1], [1.1, 0.4, -0.2]])
Sigma_w = np.eye(n)
S1 = np.array([[0.74, -0.21, -0.64]])
S2 = np.array([[0.37, 0.86, 0.37]])
S3 = np.array([[0, 0, 1]])
Sigma_S1 = np.array([[0.1**2]])
Sigma_S2 = np.array([[0.1**2]])
Sigma_S3 = np.array([[0.1**2]])

Sigma_0 = np.eye(n)

T = 50
s1_trace = []
s2_trace = []
s3_trace = []

# same sensor version code provided as an example
Sigma = Sigma_0
for t in range(T):
    Sigma = time_and_meas_update(Sigma, A, Sigma_w, S1, Sigma_S1)
    s1_trace.append(np.trace(Sigma))

Sigma = Sigma_0
for t in range(T):
    Sigma = time_and_meas_update(Sigma, A, Sigma_w, S2, Sigma_S2)
    s2_trace.append(np.trace(Sigma))

Sigma = Sigma_0
for t in range(T):
    Sigma = time_and_meas_update(Sigma, A, Sigma_w, S3, Sigma_S3)
    s3_trace.append(np.trace(Sigma))



plt.figure()
plt.plot(s1_trace, 'b', label='Sigma_S1')
plt.plot(s2_trace,'g', label='Sigma_S2')
plt.plot(s3_trace,'r', label='Sigma_S3')
plt.legend()
plt.show()


Sigma = Sigma_0
s123_trace = []
for t in range(T):
    '''Your code here'''
    # you need to write the round robin algo to choose which Sigma to use
    # note C is selected from Sigma_S*
    if t % n == 0:
        C = S1
        R = Sigma_S1
    elif t % n == 1:
        C = S2
        R = Sigma_S2
    else:
        C = S3
        R = Sigma_S3
    '''Your code end'''

    Sigma = time_and_meas_update(Sigma, A, Sigma_w, C, R)
    s123_trace.append(np.trace(Sigma))

plt.figure()
plt.plot(s123_trace, '--')
plt.show()

# greedy:

Sigma = Sigma_0
s1_greedy_trace = []
s2_greedy_trace = []
s3_greedy_trace = []
sgreedy_choice = []
sgreedy_trace = []
for t in range(T):
    C = S1
    R = Sigma_S1
    Sigma_try1 = time_and_meas_update(Sigma, A, Sigma_w, C, R)
    s1_greedy_trace.append(np.trace(Sigma_try1))

    C = S2
    R = Sigma_S2
    Sigma_try2 = time_and_meas_update(Sigma, A, Sigma_w, C, R)
    s2_greedy_trace.append(np.trace(Sigma_try2))

    C = S3
    R = Sigma_S3
    Sigma_try3 = time_and_meas_update(Sigma, A, Sigma_w, C, R)
    s3_greedy_trace.append(np.trace(Sigma_try3))
    '''Your code here'''
    # your greedy algorithm
    # select your Sigma based on tries!
    greedy_choice = np.argmin([s1_greedy_trace[t], s2_greedy_trace[t], s3_greedy_trace[t]])
    if greedy_choice == 0:
        Sigma = Sigma_try1
    elif greedy_choice == 1:
        Sigma = Sigma_try2
    else:
        Sigma = Sigma_try3
    sgreedy_trace.append(np.trace(Sigma))
    '''Your code end'''

plt.figure()
plt.plot(sgreedy_trace, 'g--')
plt.show()