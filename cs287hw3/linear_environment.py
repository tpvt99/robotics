import numpy as np

class LinearEnv(object):
    def __init__(self, horizon=20, multiplier=1.):
        self.A = multiplier * 0.1 * np.array([[0.0481, -0.5049, 0.0299, 2.6544, 1.0608],
                                 [2.3846, -0.2312, -0.1260, -0.7945, 0.5279],
                                 [1.4019, -0.6394, -0.1401, 0.5484, 0.1624],
                                 [-0.0254, 0.4595, -0.0862, 2.1750, 1.1012],
                                 [0.5172, 0.5060, 1.6579, -0.9407, -1.4441]])
        self.B = np.array([[-0.7789, -1.2076],
                           [0.4299, -1.6041],
                           [0.2006, -1.7395],
                           [0.8302, 0.2295],
                           [-1.8465, 1.2780]])
        self.H = 20

        self.dx = self.A.shape[1]
        self.du = self.B.shape[1]
        self.Q = np.eye(self.dx)
        self.R = np.eye(self.du)
        self._init_state =  np.array([-1.9613, -1.3127, 0.0698, 0.0935, 1.2494])
        self.reset()

    def step(self, act):
        cost = self._state.T @ self.Q @ self._state + act.T @ self.R @ act
        state = self.A @ self._state + self.B @ act
        self._state = state.copy()
        return state, cost, False, {}

    def set_state(self, state):
        self._state = state.copy()

    def reset(self):
        self._state = self._init_state.copy()
        return self._init_state.copy()