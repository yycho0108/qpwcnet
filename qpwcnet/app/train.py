#!/usr/bin/env python3

from pathlib import Path
import tensorflow as tf
import numpy as np

from qpwcnet.core.pwcnet import build_network
from qpwcnet.data.tfrecord import get_reader
from qpwcnet.data.augment import image_augment, image_resize


class FlowMseLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        # self.scale = tf.constant(1.0 / scale)
        # self.weight = tf.constant(weight)
        # self.mse = tf.keras.losses.MeanSquaredError()
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        # if true=256, pred=8, scale=1/32
        # scale = tf.cast(y_pred.shape[1], tf.float32) / y_true.shape[1]

        numer = tf.cast(tf.shape(y_pred)[1], tf.float32)
        denom = tf.cast(tf.shape(y_true)[1], tf.float32)
        scale = numer / denom

        # NOTE(yycho0108) Scale -> (1/20) as described in original paper.
        y_true_down = 0.05 * \
            tf.image.resize(y_true, tf.shape(y_pred)[1:3]) * scale
        err_norm = tf.norm(y_true_down - y_pred, ord=2, axis=3)
        loss = tf.reduce_mean(tf.reduce_sum(err_norm, axis=(1, 2)))
        weight = 0.0003125 / (scale * scale)
        return weight * loss  # tf.pow(loss, 2)
        # return weight * tf.reduce_mean(tf.square(y_true_down - y_pred))

        #weight = 0.32 * scale
        # return weight * tf.reduce_mean(tf.square(y_true_down - y_pred))


def learning_rate():
    batch_size = 4
    lr = 0.0001
    lr_boundaries = [4000, 6000, 8000, 10000]

    # Adjust the boundaries by batch size
    # lr_boundaries = [x // (batch_size // 8) for x in lr_boundaries]
    lr_values = [lr/(2**i) for i in range(len(lr_boundaries)+1)]
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_boundaries, values=lr_values)
    return lr_scheduler


def callbacks():
    return [
        # tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='/tmp/pwc.{epoch:03d}.hdf5'),
        tf.keras.callbacks.TensorBoard(
            log_dir='/tmp/pwc-logs', profile_batch='2,66'),
    ]


def preprocess(ims, flo):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0/255.0, dtype=tf.float32)
    # return image_resize(ims, flo, (256, 512))
    # apply augmentation
    return image_augment(ims, flo, (256, 512))


@tf.function
def train_step(model, losses, optim, ims, flo):
    # print('model.losses', model.losses)
    with tf.GradientTape() as tape:
        # Compute flow predictions
        outs = model(ims)

        # loss over all flo-s
        loss = tf.constant(0.0, dtype=tf.float32)
        for o, l in zip(outs, losses):
            loss = loss + l(flo, o)

        # Finalize error with regularization/auxiliary losses
        loss = loss + sum(model.losses)

    # tf.debugging.check_numerics(
    #    loss, "nan-loss"
    # )
    grads = tape.gradient(loss, model.trainable_variables)
    # for g, v in zip(grads, model.trainable_variables):
    #    tf.debugging.check_numerics(
    #        g, "nan-grad@" + v.name
    #    )
    # grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) for g in grads]
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return optim.iterations, loss


def main():
    # eager mode is just a recipe for uncompiled disasters...
    # tf.compat.v1.disable_eager_execution()

    # tf.profiler.experimental.start('/tmp/pwc-logs')
    #physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    batch_size = 4

    glob_pattern = '/media/ssd/datasets/sintel-processed/shards/sintel-*.tfrecord'
    filenames = tf.data.Dataset.list_files(glob_pattern).shuffle(32)
    # dataset = get_reader(filenames).shuffle(buffer_size=1024).repeat().batch(8)
    # dataset = get_reader(filenames).batch(8).repeat()
    dataset = (get_reader(filenames)
               .shuffle(buffer_size=32)
               .map(preprocess)
               .batch(batch_size, drop_remainder=True)
               .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
    # dataset = dataset.take(1).cache()
    model = build_network(train=True)

    # Create losses, flow mse
    losses = []
    for out in model.outputs:
        losses.append(FlowMseLoss())

    if True:
        # Custom training loop.

        # Setup directory structure.
        root = Path('/tmp/pwc')
        run_root = root / 'run'
        run_root.mkdir(parents=True, exist_ok=True)
        # NOTE(yycho0108): Automatically computing run id.
        run_id = len(list(run_root.iterdir()))
        run_dir = run_root / '{:03d}'.format(run_id)
        log_dir = run_dir / 'log'
        ckpt_dir = run_dir / 'ckpt'

        # Setup metrics.
        metrics = {}
        metrics['loss'] = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)

        # Setup handlers for training/logging.
        lr = learning_rate()
        optim = tf.keras.optimizers.Adam(learning_rate=lr)
        writer = tf.summary.create_file_writer(str(log_dir))
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=optim, net=model)
        ckpt_mgr = tf.train.CheckpointManager(
            ckpt, str(ckpt_dir), max_to_keep=8)

        # Iterate through train loop.
        for epoch in range(100):
            # prepare epoch.
            metrics['loss'].reset_states()
            # train epoch.
            for ims, flo in dataset:
                opt_iter, step_loss = train_step(
                    model, losses, optim,  ims, flo)
                metrics['loss'].update_state(step_loss)
            # log/save.
            with writer.as_default():
                tf.summary.scalar('iter', opt_iter, step=epoch)
                tf.summary.scalar('learning_rate', lr(opt_iter), step=epoch)
                tf.summary.scalar(
                    'loss', metrics['loss'].result(), step=epoch)
            ckpt_mgr.save(epoch)
        model.save_weights(str(run_dir / 'model.h5'))

    if False:
        # The following perhaps "standard" keras training loop has been removed
        # due to a stupid memory leak.

        # Create losses, flow epe
        # ...
        # run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        # run_metadata = tf.compat.v1.RunMetadata()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate()),
            loss=non_stupid_keras_loss,
            run_eagerly=False
            # options=run_options,
            # run_metadata=run_metadata
        )
        model.fit(dataset,
                  epochs=100000 // 131,
                  steps_per_epoch=131,
                  callbacks=callbacks()
                  )


if __name__ == '__main__':
    main()
