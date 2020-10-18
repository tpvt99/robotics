import numpy as np
import scipy
import scipy.signal
import json
import tensorflow as tf


def extract(x, *keys):
    """
    Args:
        x (dict or list): dict or list of dicts

    Returns:
        (tuple): tuple with the elements of the dict or the dicts of the list
    """
    if isinstance(x, dict):
        return tuple(x[k] for k in keys)
    elif isinstance(x, list):
        return tuple([xi[k] for xi in x] for k in keys)
    else:
        raise NotImplementedError

def normalize_advantages(advantages):
    """
    Args:
        advantages (np.ndarray): np array with the advantages

    Returns:
        (np.ndarray): np array with the advantages normalized
    """
    return (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)

def shift_advantages_to_positive(advantages):
    return (advantages - np.min(advantages)) + 1e-8

def discount_cumsum(x, discount):
    """
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering

    Returns:
        (float) : y[t] - discount*y[t+1] = x[t] or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    """
    """ YOUR CODE HERE FOR PROBLEM 1A """
    # hint you can also try use other functions, or figure out which "1" is actually the discount in dummy_cumsum
    result =
    """ YOUR CODE ENDS """
    return result

def set_seed(seed):
    """
    Set the random seed for all random number generators

    Args:
        seed (int) : seed to use

    Returns:
        None
    """
    import random
    import tensorflow as tf
    seed %= 4294967294
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print('using seed %s' % (str(seed)))


class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)
