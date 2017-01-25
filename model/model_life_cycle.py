import os
import tensorflow as tf

logger = tf.logging


class ModelLifeCycle(object):
    """The initialization, summarization and saving of a model"""

    def __init__(self,
                 session,
                 base_dir,
                 run_name,
                 model_id=None):
        """
        Args:
            session: TFSession object
                The tensorflow session object
            base_dir: string
                The root directory for model execution
            run_name: string
                The name of the run
            model_id: string, optional
                Indicates initialization from a previously-saved model
        """
        self._session = session
        self._base_dir = base_dir
        self._run_name = run_name

        # The saver is responsible for serializing models/parameters
        self._saver = tf.train.Saver(tf.global_variables())

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
            self._saver.restore(self._session,
                                self._checkpoint_prefix + model_id)
        else:
            self._session.run(tf.global_variables_initializer())

    def summarize(self, name, summary, current_step):
        """Utility function to write summaries.

        Args:
            global_step: int
                Global step of the training procedure
        """

        # Create directory and writer if it does not exist
        if not hasattr(self, '_summary_writer' + name):
            summary_dir = os.path.join(self._out_dir, "summaries", name)
            setattr(self,
                    '_summary_writer' + name,
                    tf.summary.FileWriter(summary_dir, self._session.graph))

        getattr(self, '_summary_writer' + name).add_summary(summary,
                                                            current_step)

    def save(self, global_step):
        """This serializes the current model, i.e., learned parameter values.

        Args:
            global_step: int
                Global step of the training procedure
        """
        path = self._saver.save(self._session, self._checkpoint_prefix,
                                global_step=global_step)
        logger.info("Saved model checkpoint to {}\n".format(path))
