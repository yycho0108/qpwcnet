#!/usr/bin/env python3

import tensorflow as tf


class TensorBoardFlowImage(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: str, tag: str):
        super().__init__()
        self.writer = tf.summary.create_file_writer(
            '{}/{}'.format(log_dir, 'flow'))
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        self.writer.add_summary(summary, epoch)
