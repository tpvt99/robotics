import tensorflow as tf
import numpy as np


def normalize(data_array, mean, std):
    """how to do normalization"""
    """ YOUR CODE HERE FOR PROBLEM 2A.3"""
    # hint: remember to fix numerical issues, such as divided by 0!
    result = None
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


def create_dnn(name,
               output_dim,
               kernel_sizes,
               strides,
               num_filters,
               hidden_nonlinearity,
               output_nonlinearity,
               hidden_dim,
               n_channels,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (tuple): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None
    assert len(kernel_sizes) == len(strides) == len(num_filters)

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')
    with tf.variable_scope(name):
        x = input_var

        x = tf.reshape(x, [-1, hidden_dim, hidden_dim, n_channels])

        for idx, (kernel_size, stride, filter) in enumerate(zip(kernel_sizes, strides, num_filters)):
            x = tf.layers.conv2d_transpose(x,
                                           filters=filter,
                                           kernel_size=kernel_size,
                                           strides=stride,
                                           name='conv_t_%d' % idx,
                                           activation=hidden_nonlinearity,
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init,
                                           )

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x,
                            np.prod(output_dim),
                            name='output',
                            activation=output_nonlinearity,
                            kernel_initializer=w_init,
                            bias_initializer=b_init,
                            # reuse=reuse,
                            )

        output_var = tf.reshape(x, (-1,) + output_dim)

    return input_var, output_var


def create_mlp(name,
               output_dim,
               hidden_sizes,
               hidden_nonlinearity,
               output_nonlinearity,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               batch_normalization=False,
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')
    x = input_var

    for idx, hidden_size in enumerate(hidden_sizes):
        if batch_normalization == 'traning':
            x = tf.layers.batch_normalization(x, training=True)
        elif batch_normalization == 'testing':
            x = tf.layers.batch_normalization(x, training=False)

        x = tf.layers.dense(x,
                            hidden_size,
                            name='hidden_%d' % idx,
                            activation=hidden_nonlinearity,
                            kernel_initializer=w_init,
                            bias_initializer=b_init,
                            )

    if batch_normalization == 'traning':
        x = tf.layers.batch_normalization(x, training=True)
    elif batch_normalization == 'testing':
        x = tf.layers.batch_normalization(x, training=False)

    output_var = tf.layers.dense(x,
                                 output_dim,
                                 name='output',
                                 activation=output_nonlinearity,
                                 kernel_initializer=w_init,
                                 bias_initializer=b_init,
                                 )

    return input_var, output_var
