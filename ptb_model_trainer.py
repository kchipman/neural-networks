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
    "base_dir", ".",
    "The directory to which experimental results are written")

tf.flags.DEFINE_integer("batch_size", 20, "Batch Size")
tf.flags.DEFINE_integer("vocab_size", 10000, "Size of the vocabulary")
tf.flags.DEFINE_integer("num_steps", 35, "Number of time steps")

# Model parameters
tf.flags.DEFINE_boolean("bidirectional", False, "Bidirectional")
tf.flags.DEFINE_integer(
    "rnn_layers", 3,
    "The number of layers in the recurrent neural network ")
tf.flags.DEFINE_integer(
    "num_units", 100, "Number of hidden units in an RNN cell")
tf.flags.DEFINE_string("cell_type", "GRU", "Type of cell")

# For training only
tf.flags.DEFINE_float("keep_probability", 1.0, "Dropout keep probability")
tf.flags.DEFINE_string("clipping", None, "Perform gradient clipping")
tf.flags.DEFINE_float("base_lr", 0.01, "Initial learning rate")
tf.flags.DEFINE_float(
    "lr_decay_factor", 0.96,
    "The decay rate of the learning rate")
tf.flags.DEFINE_integer("num_epochs", 55, "Total number of training epochs")
tf.flags.DEFINE_integer(
    "num_epochs_per_decay", 1,
    "Every number of epochs before learning rate decays")

# Parameters related to tracking the training procedure
tf.flags.DEFINE_integer(
    "checkpoint_every", 250,
    "Save model after this many steps")

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

    session = tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement
        )
    )

    raw_data = ptb_reader.ptb_raw_data(FLAGS.data_directory)
    train_data, valid_data, test_data, _ = raw_data
    input_data, targets = ptb_reader.ptb_producer(
        train_data, FLAGS.batch_size, FLAGS.num_steps)

    # Infer the number of batches in one epoch
    num_batches = ((len(train_data) // FLAGS.batch_size) - 1) \
        // FLAGS.num_steps
    # Infer number of iterations per learning rate decay
    num_iterations_per_decay = num_batches * FLAGS.num_epochs_per_decay

    # Define the connectionist temporal classification model
    model = PTBModel(
        input_data=input_data,
        labels=targets,
        num_units=FLAGS.num_units,
        vocab_size=FLAGS.vocab_size,
        rnn_layers=FLAGS.rnn_layers,
        cell_type=FLAGS.cell_type,
        keep_probability=FLAGS.keep_probability,
        bidirectional=FLAGS.bidirectional
    )

    # Define the model
    global_step = tf.Variable(0, name="global_step", trainable=False)

    train_op = _training_operation(
        model.loss, global_step, num_iterations_per_decay)
    loss_summary = tf.scalar_summary(
            "batch-wise mean perplexity", model.loss)

    # Train Summaries
    operations = {
        'train_op': train_op,
        'global_step': global_step,
        'loss': model.loss,
        'summary': loss_summary
    }

    model_lc = ModelLifeCycle(
        model,
        session,
        FLAGS.base_dir,
        FLAGS.run_name)

    # Initialize the model, optionally from a previously saved model
    model_lc.initialize(model_id=FLAGS.model_id)

    # This is required for the data input pipeline
    tf.train.start_queue_runners(sess=session)

    # Train the model using all but the first batch
    _train(session=session,
           operations=operations,
           model_lc=model_lc,
           num_epochs=FLAGS.num_epochs,
           num_batches=num_batches,
           num_steps=FLAGS.num_steps,
           checkpoint_every=FLAGS.checkpoint_every)


def _train(session,
           operations,
           model_lc,
           num_epochs,
           num_batches,
           num_steps,
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

        checkpoint_every: int
            Save this state of the current model every this interval
    """

    # Iterate over epochs
    for e in range(num_epochs):
        costs = 0.0
        iters = 0

        # Iterate over batches within the same epoch
        for b in range(num_batches):
            results = session.run(operations)
            step = results['global_step']

            model_lc.summarize(
                "train", results['summary'], results['global_step'])

            costs += results['loss']
            iters += num_steps

            # Save model if at a checkpoint
            if step > 0 and step % FLAGS.checkpoint_every == 0:
                model_lc.save(operations['global_step'])

            logger.info(
                " Iteration: {i}, epoch: {e}, step: {step}, "
                "perplexity: {loss}".format(
                    i=e * num_batches + b,
                    e=e,
                    step=results['global_step'],
                    loss=np.exp(costs / iters)))


def _training_operation(loss_function, global_step, num_iterations_per_decay):
    """Decay the learning rate exponentially based on the number of steps"""

    lr = tf.train.exponential_decay(FLAGS.base_lr,
                                    global_step,
                                    num_iterations_per_decay,
                                    FLAGS.lr_decay_factor,
                                    staircase=True)

    optimizer = tf.train.AdamOptimizer(lr)

    if FLAGS.clipping is None:
        train_op = optimizer.minimize(
            loss_function,
            global_step=global_step)
    else:
        gradients = optimizer.compute_gradients(loss_function)
        clipped_gradients = [
            (tf.clip_by_value(grad, -1., 1.), var)
            for grad, var in gradients
        ]
        train_op = optimizer.apply_gradients(
            clipped_gradients, global_step=global_step)

    return train_op


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()
