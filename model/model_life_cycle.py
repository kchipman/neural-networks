import os
import tensorflow as tf

logger = tf.logging


class ModelLifeCycle(object):
    """The initialization, summarization and saving of a model"""

    def __init__(self,
                 model,
                 session,
                 base_dir,
                 run_name):
        """
        Args:
            model: arbitrary class
                An arbitrary neural network model
            session: TFSession object
                The tensorflow session object
            base_dir: string
                The root directory for model execution
            run_name: string
                The name of the run
        """

        self._model = model
        self._session = session
        self._base_dir = base_dir
        self._run_name = run_name

    def initialize(self, model_id=None):
        """This function initializes the model, either with random values for
        all tradable parameters, or by restoring from a previously-saved model.

        Args:
            model_id: string, optional
                Indicates initialization from a previously-saved model
        """

        # The saver is responsible for serializing models/parameters
        self._saver = tf.train.Saver(tf.all_variables())

        self._out_dir = os.path.abspath(
            os.path.join(self._base_dir, "runs", self._run_name))

        checkpoint_dir = os.path.abspath(
            os.path.join(self._out_dir, "checkpoints"))
        self._checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        # Create directory if it does not exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Either restore from previous model, or initialize anew
        if model_id is not None:
            logger.info('restore ' + model_id)
            self._saver.restore(
                self._session, self._checkpoint_prefix + model_id)
        else:
            self._session.run(tf.initialize_all_variables())

    def summarize(self, name, summary, current_step):
        if not hasattr(self, '_summary_writer'):

            summary_dir = os.path.join(self._out_dir, "summaries", name)
            self._summary_writer = tf.train.SummaryWriter(
                summary_dir, self._session.graph)

        self._summary_writer.add_summary(summary, current_step)

    def save(self, global_step):
        """This serializes the current model, i.e., learned parameter values.

        Args:
            global_step: int
                Global step of the training procedure
        """

        path = self._saver.save(
            self._session, self._checkpoint_prefix, global_step=global_step
        )
        logger.info("Saved model checkpoint to {}\n".format(path))
