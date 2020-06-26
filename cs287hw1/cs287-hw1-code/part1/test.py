# import numpy as np
#
# values = np.random.random(size=(100,)) * -100000
# transitions = np.random.random(size=(100,3,100))* 100000
# rewards = np.random.random(size=(100,3,100))* -100000
#
# values = np.load("values.npy", allow_pickle=True)
# transitions = np.load("transitions.npy", allow_pickle=True)
# discount = np.load("discount.npy", allow_pickle=True)
# rewards = np.load("rewards.npy", allow_pickle=True)
#
# q = np.multiply(transitions,rewards + discount * values).sum(axis=2)
# v = q.max(axis=1)
#
# new_policy = np.zeros(shape=transitions.shape[0:2])
# new_values = np.zeros_like(values)
# for state in range(new_values.shape[0]):
#     maxValue = None
#     for next_action in range(new_policy.shape[1]):
#         tempValue = np.sum(np.multiply(transitions[state, next_action], rewards[state, next_action] + discount * values))
#         if maxValue == None:
#             maxValue = tempValue
#         else:
#             if tempValue > maxValue:
#                 maxValue = tempValue
#     new_values[state] = maxValue
#
# for state in range(new_values.shape[0]):
#     actionValue = []
#     for next_action in range(new_policy.shape[1]):
#         # tempValue = 0
#         # for next_state in range(self.env.observation_space.n):
#         # tempValue += self.transitions[state, next_action, next_state] * \
#         #        (self.rewards[state, next_action, next_state] + self.discount * state_values[next_state])
#         tempValue = np.sum(np.multiply(transitions[state, next_action],
#                                        rewards[state, next_action] + discount * values))
#         actionValue.append(tempValue)
#     new_policy[state, np.argmax(np.array(actionValue))] = 1
#
# p = q.argmax(axis=1)
# pass

import math
for i in range(8):
    z = i
    for dim in range(3):
        t = (int) (z % 2)
        print(t)
        z = math.floor(z/2)
    print("---")