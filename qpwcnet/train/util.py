#!/usr/bin/env python3

import tensorflow as tf

from qpwcnet.core.vis import flow_to_image


class TensorBoardFlowImage(tf.keras.callbacks.Callback):
    def __init__(self, data, data_format, log_dir: str, tag: str):
        super().__init__()
        self.data_format = data_format
        self.data = data  # dataset? not dataset?
        self.writer = tf.summary.create_file_writer(
            '{}/{}'.format(log_dir, 'flow'))
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        flows = self.model(self.data, training=False)
        flow_images = flow_to_image(flows, data_format=self.data_format)

        with self.writer.as_default():
            tf.summary.image('flow', img, step=epoch)
        self.writer.add_summary(summary, epoch)
