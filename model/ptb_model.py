from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell import DropoutWrapper

import tensorflow as tf

logger = tf.logging


class PTBModel(object):
    """RNN based language model designed for the PTB data"""

    def __init__(self,
                 input_data,
                 labels,
                 num_units,
                 vocab_size,
                 rnn_layers=1,
                 cell_type='GRU',
                 keep_probability=1.0,
                 bidirectional=False):
        """
        Args:
            input_data: a 2-D Tensor of shape [batch_size, num_steps]
                Each element is of integer type, i.e., a word ID.
            labels: a 2-D Tensor of shape [batch_size, num_steps]
                Each element is of integer type, i.e., a word ID, but this
                Tensor is time-shifted to the right by one compared
                to the above input_data.
            num_units: int
                The number of hidden layers
            vocab_size: int
                The size of the vocabulary
            rnn_layers: int, optional, default to 1
                 The number of RNN layers
            cell_type: string, optional, default to 'GRU'
                Definition of the RNN; can be plain-vanilla, or other type
                supported by tensorflow; choices: ['GRU', 'LSM', 'Basic']
            keep_probability: float, optional, default to 1.0
                The keep probability used in drop-out
            bidirectional: boolean, optinal, default to False
                Indicates bidirectional or unidirectional
        """

        embedding = tf.get_variable(
            "embedding", [vocab_size, num_units], dtype=tf.float32)
        rnn_layer_inputs = tf.nn.embedding_lookup(embedding, input_data)

        outputs = _get_rnn_layer(cell_type,
                                 num_units,
                                 bidirectional,
                                 rnn_layers,
                                 inputs=rnn_layer_inputs,
                                 keep_probability=keep_probability)

        # Bi-directionality doubles the number of parameters
        num_units_total = num_units
        if bidirectional:
            num_units_total = num_units * 2

        # Covert Tensor [batch_size, max_timesteps, num_units]
        # to Tensor [batch_size * max_timesteps, num_units]
        outputs = tf.reshape(outputs, [-1, num_units_total])

        # Weights
        W_out = tf.Variable(tf.truncated_normal(
            [num_units_total, vocab_size], stddev=0.1))
        # Biases
        b = tf.Variable(tf.zeros([vocab_size]))
        logits = tf.matmul(outputs, W_out) + b

        # The following returns a 1D batch_size * num_steps float Tensor:
        # the log-perplexity for each sequence.
        seq_log_perps = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(labels, [-1])],
            [tf.ones([logits.get_shape()[0].value], dtype=tf.float32)],
            average_across_timesteps=True)

        self.loss = tf.reduce_sum(seq_log_perps) \
            / input_data.get_shape()[0].value  # Averaged by batch_size


def _get_rnn_layer(cell_type,
                   num_units,
                   bidirectional,
                   rnn_layers,
                   inputs,
                   keep_probability):
    """This function encapsulates the creation of one or more RNN layers. The
    specific type of cell is parameterized, and can be from any of the
    supported tensorflow cell types.

    Args:
        cell_type: string
            Definition of the RNN; can be plain-vanilla, or other type
            supported by tensorflow; choices: ['GRU', 'LSM', 'Basic']
        num_units: int
            The number of hidden units
        bidirectional: boolean
            This indicates bidirectional or unidirectional.
        rnn_layers: int
            The number of RNN layers
        inputs: float Tensor, shape=([batch_size, sequence_length,
            num_features])
            The input data is a three-dimensional Tensor.
        keep_probability: float
            The keep probability used in drop-out

    Returns:
        outputs: float Tensor, shape=([batch_size, max_timesteps, num_units])
            The output data is a three-dimensional Tensor. Note: the
            bidirectional function will return a tuple of 2 tensors, one each
            for the forward and reverse directions.
    """

    cell_fw = _gen_rnn_cell(
        cell_type, num_units, rnn_layers, keep_probability)

    # The first variable returned contains both forward and reverse values
    if bidirectional:
        cell_bw = _gen_rnn_cell(
            cell_type, num_units, rnn_layers, keep_probability)
        outputs, _ = bidirectional_dynamic_rnn(cell_fw,
                                               cell_bw,
                                               inputs,
                                               dtype=tf.float32)
    else:
        outputs, _ = dynamic_rnn(cell_fw,
                                 inputs,
                                 dtype=tf.float32)
    return outputs


def _gen_rnn_cell(cell_type,
                  num_units,
                  rnn_layers,
                  keep_probability):
    """This generates an RNN cell according to specifications.

    Args:
        cell_type: string
            Definition of the RNN; can be plain-vanilla, or other type
            supported by tensorflow; choices: ['GRU', 'LSM', 'Basic']
        num_units: int
            The number of hidden layers
        rnn_layers: int
            The number of RNN layers
        keep_probability: float
            The keep probability used in drop-out

    Returns:
        cell: an initialized RNN cell
    """

    # Define the RNN cell type
    if cell_type == 'GRU':
        cell = rnn_cell.GRUCell(num_units)
    elif cell_type == 'LSTM':
        cell = rnn_cell.LSTMCell(
            num_units,
            use_peepholes=True,
            state_is_tuple=True)
    else:
        cell = rnn_cell.BasicRNNCell(num_units)

    if keep_probability < 1.0:
        cell = DropoutWrapper(cell, output_keep_prob=keep_probability)

    # Multi-layer RNNs
    if rnn_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [cell] * rnn_layers, state_is_tuple=True)

    return cell
