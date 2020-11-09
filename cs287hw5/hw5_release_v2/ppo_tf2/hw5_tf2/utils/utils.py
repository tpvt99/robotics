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

def dummy_cumsum(x, discount=1):
    """
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering

    Returns:
        (float) : y[t] - discount*y[t+1] = x[t] or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    """
    return scipy.signal.lfilter([1], [1, float(-1)], x[::-1], axis=0)[::-1]

def discount_cumsum(x, discount):
    """
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering

    Returns:
        (float) : y[t] - discount*y[t+1] = x[t] or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    """
    """ YOUR CODE HERE FOR PROBLEM 1A """
    # hint you can also try use other functions, or figure out which "1" is actually the discount in dummy_cumsum
    result = scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
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

def remove_scope_from_name(name, scope):
    """
    Args:
        name (str): full name of the tf variable with all the scopes. Ex: name = 'policy/mean_network/output/bias:0'
        will become 'mean_network/output/bias' given scope = 'policy'

    Returns:
        (str): full name of the variable with the scope removed
    """
    result = name.split(scope)[1]
    result = result[1:] if result[0] == '/' else result
    return result.split(":")[0]

def concat_tensor_dict_list(tensor_dict_list, end=None):
    """
    Args:
        tensor_dict_list (list) : list of dicts of lists of tensors

    Returns:
        (dict) : dict of lists of tensors
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = np.concatenate([x[k][:end] for x in tensor_dict_list])
        ret[k] = v
    return ret


def stack_tensor_dict_list(tensor_dict_list, max_path=None, end=None):
    """
    Args:
        tensor_dict_list (list) : list of dicts of tensors

    Returns:
        (dict) : dict of lists of tensors
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            if max_path is not None:
                v = np.asarray([
                                np.concatenate([x[k], np.zeros((max_path - x[k].shape[0],) + x[k].shape[1:])])
                                for x in tensor_dict_list])
            else:
                v = np.asarray([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret

class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)
