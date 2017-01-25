from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import DropoutWrapper

import tensorflow as tf

logger = tf.logging


class PTBModel(object):
    """RNN based language model designed for the PTB data"""

    def __init__(self,
                 input_data,
                 targets,
                 num_units,
                 vocab_size,
                 rnn_layers,
                 keep_probability):
        """
        Args:
            input_data: a 2-D Tensor of shape [batch_size, num_steps]
                Each element is of integer type, i.e., a word ID.
            targets: a 2-D Tensor of shape [batch_size, num_steps]
                Each element is of integer type, i.e., a word ID, but this
                Tensor is time-shifted to the right by one compared
                to the above input_data.
            num_units: int
                The number of hidden layers
            vocab_size: int
                The size of the vocabulary
            rnn_layers: int, optional, default to 1
                 The number of RNN layers
            keep_probability: float, optional, default to 1.0
                The keep probability used in drop-out
        """

        embedding = tf.get_variable(
            "embedding", [vocab_size, num_units], dtype=tf.float32)

        rnn_inputs = tf.nn.embedding_lookup(embedding, input_data)
        if keep_probability < 1.0:
            rnn_inputs = tf.nn.dropout(rnn_inputs, keep_probability)

        outputs, cell_fw, new_states = _get_rnn_layer(num_units, rnn_layers,
                                                      rnn_inputs,
                                                      keep_probability)

        batch_size = input_data.get_shape()[0].value
        num_steps = input_data.get_shape()[1].value

        # Operation to update the RNN states for the subsequent batch, using the
        # states corresponding to the final frame of the just-completed batch.
        self.update_op = _get_state_update_op(new_states, batch_size, cell_fw)

        # Convert Tensor [batch_size, max_timesteps, num_units]
        # to Tensor [batch_size * max_timesteps, num_units]
        outputs = tf.reshape(outputs, [-1, num_units])

        # Weights for the output projection
        W_out = tf.get_variable("W_out", [num_units, vocab_size],
            initializer=tf.truncated_normal_initializer(stddev=0.1))

        # Biases for the output projection
        b_out = tf.get_variable("b_out",
            initializer=tf.zeros_initializer([vocab_size]))

        logits = tf.matmul(outputs, W_out) + b_out

        # The following returns a 1D batch_size * num_steps float Tensor:
        # the log-perplexity for each sequence.
        seq_log_perps = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(targets, [-1])],
            [tf.ones([logits.get_shape()[0].value], dtype=tf.float32)],
            average_across_timesteps=True)

        self.loss = tf.reduce_sum(seq_log_perps) / batch_size

        self.perplexity = tf.exp(self.loss/num_steps)


def _get_rnn_layer(num_units,
                   rnn_layers,
                   inputs,
                   keep_probability):
    """This function encapsulates the creation of one or more RNN layers.

    A fundamental LSTM cell is optionally wrapped in a dropout wrapper.

    Args:
        num_units: int
            The number of hidden units
        rnn_layers: int
            The number of RNN layers
        inputs: float Tensor, shape=([batch_size, sequence_length,
            num_features])
            The input data is a three-dimensional Tensor.
        keep_probability: float
            The keep probability used in drop-out

    Returns:
        outputs: float Tensor, shape=([batch_size, max_timesteps, num_units])
            The output data is a three-dimensional Tensor.
        cell_fw: LSTM Cell
            The forward cell
        new_states: Tensor
            The final states from the forward cell

    """
    cell_fw = _gen_rnn_cell(num_units, rnn_layers, keep_probability)

    outputs, new_states = dynamic_rnn(cell_fw, inputs, dtype=tf.float32)

    return outputs, cell_fw, new_states


def _gen_rnn_cell(num_units, rnn_layers, keep_probability):
    """This generates an RNN cell according to parameters, which optionally
    wrap the fundamental cell with a dropout wrapper, and/or a multi-cell
    wrapper.

    Args:
        num_units: int
            The number of hidden units
        rnn_layers: int
            The number of RNN layers
        keep_probability: float
            The keep probability used in drop-out

    Returns:
        cell: an initialized RNN cell
       """
    cell = rnn_cell.LSTMCell(
        num_units, use_peepholes=True, state_is_tuple=True)

    if keep_probability < 1.0:
        cell = DropoutWrapper(cell, output_keep_prob=keep_probability)

    # Multi-layer RNNs
    if rnn_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [cell] * rnn_layers, state_is_tuple=True)

    return cell

###############################################################################
# From:
# http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
#
# NOTE**: this approach, which relies on non-trainable variables, is not thread
# safe.
#
def _get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(batch_size, tf.float32):
        state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))

    # Return as a tuple, so that it can be fed as an initial state
    return tuple(state_variables)


def _get_state_update_op(new_states, batch_size, cell):

    state_variables = _get_state_variables(batch_size, cell)

    # Add an operation to update the train states with the last state tensors
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)

