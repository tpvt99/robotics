import tensorflow as tf
import numpy as np

def normalize(data_array, mean, std):
    """how to do normalization"""
    """ YOUR CODE HERE FOR PROBLEM 2A.3"""
    # hint: remember to fix numerical issues, such as divided by 0!
    result = (data_array - mean) / (std + 1e-8)
    """ YOUR CODE ENDS"""
    return result


def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean

def train_test_split(obs, act, delta, test_split_ratio=0.2):
    assert obs.shape[0] == act.shape[0] == delta.shape[0]
    dataset_size = obs.shape[0]
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(dataset_size * (1-test_split_ratio))

    idx_train = indices[:split_idx]
    idx_test = indices[split_idx:]
    assert len(idx_train) + len(idx_test) == dataset_size

    return obs[idx_train, :], act[idx_train, :], delta[idx_train, :], \
           obs[idx_test, :], act[idx_test, :], delta[idx_test, :]