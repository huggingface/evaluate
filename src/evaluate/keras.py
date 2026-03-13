import os

import tensorflow as tf
import tensorflow.keras as keras


class KerasCallback(keras.callbacks.Callback):
    def __init__(
        self,
        model,
        model_inputs,
        metric,
        metric_inputs,
        model_output_name="predictions",
        predictions_processor=None,
        log_dir=None,
    ):
        self.model = model
        self.model_inputs = model_inputs
        self.model_output_name = model_output_name
        self.predictions_processor = predictions_processor
        self.metric = metric
        self.metric_inputs = metric_inputs
        self.epoch = 0
        self.predictions_processor = predictions_processor

        if log_dir is not None:
            self.summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, model.name))
        else:
            self.summary_writer = None

    def on_epoch_end(self, batch, logs=dict()):
        self.epoch += 1
        predictions = self.model.predict(self.model_inputs)
        if self.predictions_processor is not None:
            predictions = self.predictions_processor(predictions)
        self.metric_inputs.update({self.model_output_name: predictions})

        result = self.metric.compute(**self.metric_inputs)
        logs.update(result)
        if self.summary_writer is not None:
            self._write_metric(result)

    def _write_metric(self, result):
        with self.summary_writer.as_default():
            for name, value in result.items():
                tf.summary.scalar(
                    name,
                    value,
                    step=self.epoch,
                )
            self.summary_writer.flush()
