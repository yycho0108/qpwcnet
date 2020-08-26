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


class FlowMseLossFineTune(tf.keras.losses.Loss):
    def __init__(self, q=0.4, eps=0.01, *args, **kwargs):
        self.q = q
        self.eps = eps
        self.config_ = {
            'q': q,
            'eps': eps
        }
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
        err_norm = tf.norm(y_true_down - y_pred, ord=1, axis=3)
        err_norm = tf.pow(err_norm + self.eps, self.q)
        loss = tf.reduce_mean(tf.reduce_sum(err_norm, axis=(1, 2)))
        weight = 0.0003125 / (scale * scale)
        return weight * loss  # tf.pow(loss, 2)

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def learning_rate(batch_size):
    base_lr = 0.0001
    # Fixed lr boundaries, by number of training samples (not number of batches).
    lr_boundaries = [128000, 192000, 256000, 320000]
    # Adjust the boundaries by batch size.
    lr_boundaries = [int(x/batch_size) for x in lr_boundaries]

    lr_values = [base_lr/(2**i) for i in range(len(lr_boundaries)+1)]
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


def preprocess_passthrough(ims, flo):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0/255.0, dtype=tf.float32)
    # resize, no augmentation.
    ims, flo = image_resize(ims, flo, (256, 512))
    # 0.0-1.0 -> -0.5, 0.5
    ims = ims - 0.5
    return ims, flo


def preprocess(ims, flo):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0/255.0, dtype=tf.float32)
    # apply augmentation
    ims, flo = image_augment(ims, flo, (256, 512))
    # 0.0-1.0 -> -0.5, 0.5
    ims = ims - 0.5
    return ims, flo


@tf.function
def train_step(model, losses, optim, ims, flo):
    # print('model.losses', model.losses)
    with tf.GradientTape() as tape:
        # Compute flow predictions.
        pred_flows = model(ims)

        # Compute loss over all flow pairs at each scale.
        flow_losses = [l(flo, o) for o, l in zip(pred_flows, losses)]

        # Finalize error with regularization/auxiliary losses
        loss = sum(flow_losses) + sum(model.losses)

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
    return optim.iterations, flow_losses, loss


def main():
    # eager mode is just a recipe for uncompiled disasters...
    # tf.compat.v1.disable_eager_execution()

    # tf.profiler.experimental.start('/tmp/pwc-logs')
    #physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    batch_size = 4
    num_epoch = 600

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
        # losses.append(FlowMseLoss())
        losses.append(FlowMseLossFineTune())

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
        for out in model.outputs:
            name = 'flow-loss-{:02d}'.format(out.shape[1])
            metrics[name] = tf.keras.metrics.Mean(name=name, dtype=tf.float32)

        # Setup handlers for training/logging.
        # lr = learning_rate(batch_size)
        lr = 6.25e-6
        optim = tf.keras.optimizers.Adam(learning_rate=lr)
        writer = tf.summary.create_file_writer(str(log_dir))
        ckpt = tf.train.Checkpoint(optimizer=optim, net=model)
        ckpt_mgr = tf.train.CheckpointManager(
            ckpt, str(ckpt_dir), max_to_keep=8)

        # [Optional] load checkpoint.
        if True:
            load_ckpt = tf.train.Checkpoint(optimizer=optim, net=model)
            # load_ckpt_mgr = tf.train.CheckpointManager(
            #    load_ckpt, '/tmp/pwc/run/006/ckpt/', max_to_keep=8)
            load_ckpt_mgr = tf.train.CheckpointManager(
                load_ckpt, '/tmp/pwc/run/007/ckpt/', max_to_keep=8)
            load_ckpt.restore(load_ckpt_mgr.latest_checkpoint)

        # Iterate through train loop.
        for epoch in range(num_epoch):
            print('Epoch {:03d}/{:03d}'.format(epoch, num_epoch))
            # prepare epoch.
            for v in metrics.values():
                v.reset_states()

            # train epoch.
            for ims, flo in dataset:
                opt_iter, flow_loss, step_loss = train_step(
                    model, losses, optim,  ims, flo)
                # update metrics.
                metrics['loss'].update_state(step_loss)
                for out, l in zip(model.outputs, flow_loss):
                    name = 'flow-loss-{:02d}'.format(out.shape[1])
                    metrics[name].update_state(l)

            # log/save.
            with writer.as_default():
                tf.summary.scalar('iter', opt_iter, step=epoch)
                tf.summary.scalar('learning_rate', lr, step=epoch)
                for k, v in metrics.items():
                    tf.summary.scalar(k, v.result(), step=epoch)
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
