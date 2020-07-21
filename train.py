#!/usr/bin/env python3
import tensorflow as tf

from lib.pwcnet import build_network
from lib.tfrecord import get_reader


class FlowMseLoss(tf.keras.losses.Loss):
    def __init__(self, scale, weight, *args, **kwargs):
        self.scale = tf.constant(1.0 / scale)
        self.weight = tf.constant(weight)
        # self.mse = tf.keras.losses.MeanSquaredError()
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        with tf.name_scope('flow_mse'):
            # print(y_true.shape, y_pred.shape, self.scale)
            y_true_down = tf.image.resize(
                y_true, y_pred.shape[1:3]) * self.scale
            return self.weight * tf.reduce_mean(tf.square(y_true_down - y_pred))
            # return self.mse(y_true_down, y_pred)


def learning_rate():
    batch_size = 8
    lr = 0.0001
    lr_boundaries = [4000, 6000, 8000, 10000]

    # Adjust the boundaries by batch size
    lr_boundaries = [x // (batch_size // 8) for x in lr_boundaries]
    lr_values = [lr/(2**i) for i in range(len(lr_boundaries)+1)]
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_boundaries, values=lr_values)
    return lr_scheduler


def callbacks():
    return [
        # tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='/tmp/pwc.{epoch:03d}.hdf5'),
        tf.keras.callbacks.TensorBoard(log_dir='/tmp/pwc-logs'),
    ]


def main():
    glob_pattern = '/media/ssd/datasets/sintel-processed/shards/sintel-*.tfrecord'
    filenames = tf.data.Dataset.list_files(glob_pattern).shuffle(32)
    dataset = get_reader(filenames).shuffle(buffer_size=1024).repeat().batch(8)
    # dataset = get_reader(filenames).batch(8).repeat()
    net = build_network(train=True)

    # Create losses, flow mse
    losses = []
    scale = 1.0
    weight = 0.32
    for out in net.outputs:
        losses.append(FlowMseLoss(scale, weight))
        scale *= 2
        weight /= 2
    losses = losses[::-1]

    # Create losses, flow epe
    # ...
    net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate()),
        loss=losses
    )
    net.fit(dataset,
            epochs=10000 // 131,
            steps_per_epoch=131,
            callbacks=callbacks()
            )


if __name__ == '__main__':
    main()
