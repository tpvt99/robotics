import numpy as np
from part4 import kf_smooth
import matplotlib.pyplot as plt

x0 = np.array([[60], [60], [60]])
P_0 = np.eye(3) * 2

# I found initially overestimating Q and R gives better learning of Q and R
# during EM

'''Your code here'''
Q = np.zeros((x0.shape[0], x0.shape[0]))
R = np.array([[0.36]])# make it a 2-D array even it's just one number, such as np.array([[x]])

A = np.array([[1.02, 0, 0], [0, 1.06, 0], [0, 0, 1.11]])
B = np.zeros((3,1))
C = np.ones((1,3))
d = np.zeros((1,1))
'''Your code end'''

done_with_1 = 0
done_with_2 = 0
done_with_3 = 0

for T in range(20, 100):
    if done_with_1 and done_with_2 and done_with_3:
        break

    u = np.zeros([1, T])
    y = np.zeros([1, T])
    # For exp
    x_exp = np.zeros([3, T])
    y_exp = np.zeros([1, T])
    for t in range(T):
        if t == 0:
            x_exp[:, t] = np.random.multivariate_normal(x0.reshape(3), P_0)
            y_exp[:, t] = C @ x_exp[:, t] + np.random.multivariate_normal(np.zeros((1)), R)
        else:
            x_exp[:, t] = A @ x_exp[:, t-1] + np.random.multivariate_normal(np.zeros((3)), Q)
            y_exp[:, t] = C @ x_exp[:, t] + np.random.multivariate_normal(np.zeros((1)), R)

    xfilt, _, Vfilt, loglik, xsmooth, Vsmooth, _, _ = kf_smooth(y, A, B, C, d, u, Q, R, x0, P_0);

    ''' Your code here '''
    # you need to write the correct condition after each ``if''
    if Vsmooth[0, 0, 0] < 0.01:
        done_with_1 = 1
        print('done with 1')
        print(f'Time for U is {T}')

    if Vsmooth[1, 1, 0] < 0.01:
        done_with_2 = 1
        print('done with 2')
        print(f'Time for V is {T}')

    if Vsmooth[2, 2, 0] < 0.01:
        done_with_3 = 1
        print('done with 3')
        print(f'Time for W is {T}')
    ''' Your code end '''
    print('Xsmooth: ', xsmooth[:,:,0])
    print('Vsmooth: ', Vsmooth[:, :, 0])
    print('----')

plt.figure(figsize=(12, 6))
for i in range(3):
    plt.plot(np.squeeze(xsmooth)[i, :], '--', linewidth=1)
plt.xlabel('timestep')
plt.ylabel('state')
plt.title('KF results')
plt.show()