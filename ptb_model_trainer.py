import tensorflow as tf
import numpy as np
import time
import data_reader.ptb_reader as ptb_reader
from model.ptb_model import PTBModel
from model.model_life_cycle import ModelLifeCycle

logger = tf.logging

# These two parameters specify where data is read from,
# and where results are written to
tf.flags.DEFINE_string(
    "data_directory", "./data/ptb_data/",
    "The directory possessing training data")
tf.flags.DEFINE_string(
    "runs_dir", ".",
    "The directory to which experimental results are written")

tf.flags.DEFINE_integer("batch_size", 20, "Batch Size")
tf.flags.DEFINE_integer("vocab_size", 10000, "Size of the vocabulary")
tf.flags.DEFINE_integer("num_steps", 35, "Number of time steps")

# Model parameters
tf.flags.DEFINE_integer("rnn_layers", 3, "The number of RNN layers")
tf.flags.DEFINE_integer(
    "num_units", 100, "Number of hidden units in an RNN cell")

# For training only
tf.flags.DEFINE_float("keep_probability", 1.0, "Dropout keep probability")
tf.flags.DEFINE_float("base_lr", 0.01, "Initial learning rate")
tf.flags.DEFINE_float("lr_decay_factor", 0.96,
                      "The decay rate of the learning rate")
tf.flags.DEFINE_integer("num_epochs", 55, "Total number of training epochs")
tf.flags.DEFINE_integer("num_epochs_per_decay", 1,
                        "Every number of epochs before learning rate decays")
tf.flags.DEFINE_float("clipping", None,
                      """Threshold for (hard) gradient clipping.
                      Default equates to no clipping.""")
tf.flags.DEFINE_boolean("truncated_backprop", True, "Truncated backpropagation")

# Parameters related to tracking the training procedure
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after x steps")
tf.flags.DEFINE_integer("validate_every", 100, "Run validation every x steps")
tf.flags.DEFINE_integer("num_validation_batches", 20,
                        "Number of validation batches")

# These next parameters specify whether an existing model is used as an
# initialization point
tf.flags.DEFINE_string("run_name", str(int(time.time())),
                       "The experiment name (default: time)")
tf.flags.DEFINE_string("model_id", None,
                       "Indicates restoration of existing model.")

# Indicates that Tensor flow should output the placement of devices
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement device")

# For convenience
FLAGS = tf.flags.FLAGS


def main(_):
    # Print the current configuration to the terminal
    FLAGS._parse_flags()
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))


    raw_data = ptb_reader.ptb_raw_data(FLAGS.data_directory)
    train_data, validation_data, test_data, _ = raw_data

    # Infer the number of batches in one epoch
    num_batches = ((len(train_data) // FLAGS.batch_size) - 1) \
        // FLAGS.num_steps
    # Infer number of iterations per learning rate decay
    num_iterations_per_decay = num_batches * FLAGS.num_epochs_per_decay

    with tf.variable_scope("rnn_model"):
        model = _get_model(train_data, FLAGS.keep_probability)

    with tf.variable_scope("rnn_model", reuse=True):
        validation_model = _get_model(validation_data, keep_probability=1.0)

    session = tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement
        )
    )

    validation_runner = ValidationRunner(
        validation_model, session, FLAGS.num_validation_batches)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    # operations is a dictionary possessing training-related operations
    operations = _get_operations(global_step, model, num_iterations_per_decay)

    # The lifecycle manager may initialize the graph with previously-stored
    # parameters, specified by the "model_id" parameter
    model_lc = ModelLifeCycle(
        session,
        FLAGS.runs_dir,
        FLAGS.run_name,
        model_id=FLAGS.model_id)

    # This is required for the data input pipeline
    tf.train.start_queue_runners(sess=session)

    # Train the model
    _train(session=session,
           operations=operations,
           model_lc=model_lc,
           num_epochs=FLAGS.num_epochs,
           num_batches=num_batches,
           num_steps=FLAGS.num_steps,
           validation_runner=validation_runner,
           checkpoint_every=FLAGS.checkpoint_every)

def _get_operations(global_step, model, num_iterations_per_decay):
    """Logic for constructing the tensorflow operations.

    Args:
        global_step: Tensorflow variable
            The step counter of the training procedure
        model: Tensor
            The tensorflow graph
        num_iterations_per_decay: integer
            Number of iterations for each learning rate period
    Returns:
        operations: dict
            Dictionary possessing  ensorflow operations, for training

    """
    train_op = _training_operation(
        model.loss, global_step, num_iterations_per_decay)

    operations = {
        'train_op': train_op,
        'global_step': global_step,
        'perplexity': model.perplexity,
        'summary': tf.summary.scalar("train_perplexity", model.perplexity),
    }

    # Incorporate the truncated back propagation, if specified.
    # note that the
    if FLAGS.truncated_backprop:
        operations['update'] = model.update_op

    return operations

def _get_model(data, keep_probability):
    """Constructs the tensorflow graph.

    Args:
        data: Tensor
            The tensorflow operation representing the entry point into the
            data pipeline.
        keep_probability: float
            The dropout probability for RNN cells

    Returns:
        model: Tensor
            The tensorflow graph

    """
    # Construct the data pipeline
    input_data, targets = ptb_reader.ptb_producer(data,
                                                  FLAGS.batch_size,
                                                  FLAGS.num_steps)

    # Define the language model
    model = PTBModel(
        input_data=input_data,
        targets=targets,
        keep_probability=keep_probability,
        num_units=FLAGS.num_units,
        vocab_size=FLAGS.vocab_size,
        rnn_layers=FLAGS.rnn_layers
    )
    return model

def _train(session,
           operations,
           model_lc,
           num_epochs,
           num_batches,
           num_steps,
           validation_runner,
           checkpoint_every=1000):
    """This performs the core training operations for a model.

    Args:
        session: tf.Session object
            A Session object that encapsulates the environment in which
            the following operation objects are executed.
        operations: dictionary
            A map of operations to be executed in the current session
        model_lc: model.model_life_cycle.ModelLifeCycle object
            A ModelLifeCycle object that manages the current model
        num_epochs: int
            The total number of training epochs
        num_batches: int
            The number of batches in one epoch
        num_steps: int
            Number of time steps in a sequence
        validation_runner: python class
            Performs benchmarking on the validation data
        checkpoint_every: int
            Save this state of the current model every this interval
    """
    # Iterate over epochs
    for e in range(num_epochs):
        # Iterate over batches within an epoch
        for b in range(num_batches):
            results = session.run(operations)
            step = results['global_step']

            # Record training perplexity
            model_lc.summarize("train", results['summary'], step)

            # Calculate and record perplexity for the validation data
            if step > 0 and step % FLAGS.validate_every == 0:
                validation_summary = validation_runner.validation_perplexity()
                model_lc.summarize("validation", validation_summary, step)

            # Save model if at a checkpoint
            if step > 0 and step % FLAGS.checkpoint_every == 0:
                model_lc.save(operations['global_step'])

            logger.info(
                " Iteration: {i}, epoch: {e}, step: {step}, "
                "perplexity: {perplexity:0.2f}".format(
                    i=e * num_batches + b,
                    e=e,
                    step=step,
                    perplexity=results['perplexity'])
            )


def _training_operation(loss_function, global_step, num_iterations_per_decay):
    """Decay the learning rate exponentially based on the number of steps.
    The procedure can optionally implement hard clipping, per the "clipping"
    parameter.

    Args:
        loss_function: Tensorflow operation
            The objective function for gradient descent
        global_step: Tensorflow variable
            The step counter of the training procedure
        num_iterations_per_decay: integer
            Number of iterations for each learning rate period

    Returns:
        training_op: Tensorflow operation
            The training operation

    """
    lr = tf.train.exponential_decay(FLAGS.base_lr,
                                    global_step,
                                    num_iterations_per_decay,
                                    FLAGS.lr_decay_factor,
                                    staircase=True)

    optimizer = tf.train.AdamOptimizer(lr)

    if FLAGS.clipping is None:
        train_op = optimizer.minimize(loss_function,
                                      global_step=global_step)
    else:
        gradients = optimizer.compute_gradients(loss_function)
        clipped_gradients = [
            (tf.clip_by_value(grad, -1.*FLAGS.clipping, FLAGS.clipping), var)
            for grad, var in gradients
        ]
        train_op = optimizer.apply_gradients(clipped_gradients,
                                             global_step=global_step)

    return train_op


class ValidationRunner(object):
    """Responsible for running inference on validation data"""

    def __init__(self,
                 model,
                 session,
                 num_batches):
        """
        Args:
            model: arbitrary python class
                An arbitrary neural network model
            session: TFSession object
                The tensorflow session object
            num_batches: int
                The number of batches

        Members:
            _model: arbitrary python class
                An arbitrary neural network model
            _session: TFSession object
                The tensorflow session object
            _num_batches: int
                The number of batches
        """
        self._model = model
        self._session = session
        self._num_batches = num_batches

    def validation_perplexity(self):
        """Performs the validation over a predefined number of batches.
        This procedure returns an average over all of the individual batch
        perplexities, which themselves are averaged over each step.

        Returns:
            perplexity: Tensorflow summary
                The mean perplexity over all batches
        """
        total_perplexity = 0.0

        # Iterate over batches
        for b in range(self._num_batches):
            total_perplexity += self._session.run(self._model.perplexity)

        # Create the tensorflow summary
        return tf.core.framework.summary_pb2.Summary(
            value=[tf.core.framework.summary_pb2.Summary.Value(
                        tag="validation_perplexity",
                        simple_value=total_perplexity/self._num_batches)]
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()
