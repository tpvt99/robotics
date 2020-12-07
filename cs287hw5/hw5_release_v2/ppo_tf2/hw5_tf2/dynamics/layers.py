from hw5_tf2.utils.serializable import Serializable
from hw5_tf2.utils.utils import remove_scope_from_name
import tensorflow as tf
from collections import OrderedDict
from hw5_tf2.utils.networks.mlp import MLP as MLPNetwork

class Layer(Serializable):
    """
    A container for storing the current pre and post update policies
    Also provides functions for executing and updating policy parameters

    Note:
        the preupdate policy is stored as tf.Variables, while the postupdate
        policy is stored in numpy arrays and executed through tf.placeholders

    Args:
        obs_dim (int): dimensionality of the observation space -> specifies the input size of the policy
        action_dim (int): dimensionality of the action space -> specifies the output size of the policy
        name (str) : Name used for scoping variables in policy
        hidden_sizes (tuple) : size of hidden layers of network
        learn_std (bool) : whether to learn variance of network output
        hidden_nonlinearity (Operation) : nonlinearity used between hidden layers of network
        output_nonlinearity (Operation) : nonlinearity used after the final layer of network
    """
    def __init__(self,
                 name,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 input_var=None,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.input_var = input_var

        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        self.batch_normalization = kwargs.get('batch_normalization', False)

        self._params = None
        self._assign_ops = None
        self._assign_phs = None

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        raise NotImplementedError

    """ --- methods for serialization --- """

    def get_params(self):
        """
        Get the tf.Variables representing the trainable weights of the network (symbolic)

        Returns:
            (dict) : a dict of all trainable Variables
        """
        return self._params

    def get_param_values(self):
        """
        Gets a list of all the current weights in the network (in original code it is flattened, why?)

        Returns:
            (list) : list of values for parameters
        """
        param_values = self._params
        return param_values

    def set_params(self, policy_params):
        """
        Sets the parameters for the graph

        Args:
            policy_params (dict): of variable names and corresponding parameter values
        """
        assert all([k1 == k2 for k1, k2 in zip(self.get_params().keys(), policy_params.keys())]), \
            "parameter keys must match with variable"

        # if self._assign_ops is None:
        #     assign_ops, assign_phs = [], []
        #     for var in self.get_params().values():
        #         assign_placeholder = tf.placeholder(dtype=var.dtype)
        #         assign_op = tf.assign(var, assign_placeholder)
        #         assign_ops.append(assign_op)
        #         assign_phs.append(assign_placeholder)
        #     self._assign_ops = assign_ops
        #     self._assign_phs = assign_phs
        # feed_dict = dict(zip(self._assign_phs, policy_params.values()))
        # tf.get_default_session().run(self._assign_ops, feed_dict=feed_dict)

        for key, val in policy_params.items():
            self.get_params()[key] = val

    def __getstate__(self):
        state = {
            # 'init_args': Serializable.__getstate__(self),
            'network_params': self.get_param_values()
        }
        return state

    def __setstate__(self, state):
        # Serializable.__setstate__(self, state['init_args'])
        # tf.get_default_session().run(tf.variables_initializer(self.get_params().values()))
        self.set_params(state['network_params'])

class MLP(Layer):
    """
    Gaussian multi-layer perceptron policy (diagonal covariance matrix)
    Provides functions for executing and updating policy parameters
    A container for storing the current pre and post update policies

    Args:
        obs_dim (int): dimensionality of the observation space -> specifies the input size of the policy
        action_dim (int): dimensionality of the action space -> specifies the output size of the policy
        name (str): name of the policy used as tf variable scope
        hidden_sizes (tuple): tuple of integers specifying the hidden layer sizes of the MLP
        hidden_nonlinearity (tf.op): nonlinearity function of the hidden layers
        output_nonlinearity (tf.op or None): nonlinearity function of the output layer
        learn_std (boolean): whether the standard_dev / variance is a trainable or fixed variable
        init_std (float): initial policy standard deviation
        min_std( float): minimal policy standard deviation

    """

    def __init__(self, *args, **kwargs):
        # store the init args for serialization and call the super constructors
        Serializable.quick_init(self, locals())
        Layer.__init__(self, *args, **kwargs)

        self.build_graph()

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        # with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # build the actual policy network
        with tf.name_scope(self.name) as scope:
            self.output_var = MLPNetwork(name=scope + 'mlp/',
                                         output_dim=self.output_dim,
                                         hidden_sizes=self.hidden_sizes,
                                         hidden_nonlinearity=self.hidden_nonlinearity,
                                         output_nonlinearity=self.output_nonlinearity,
                                         input_dim=(None, self.input_dim,),
                                         batch_normalization=True,
                                         )

            # save the policy's trainable variables in dicts
            # current_scope = tf.get_default_graph().get_name_scope()

        self.output_var(tf.random.normal(shape = (1, self.input_dim)))

        # Create an OrderedDict to include mean_var and log_std
        self._params = OrderedDict([(remove_scope_from_name(var.name, self.name), var) for var in self.output_var.trainable_variables])
