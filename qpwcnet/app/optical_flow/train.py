#!/usr/bin/env python3

from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import logging

from qpwcnet.core.pwcnet import build_network
from qpwcnet.core.layers import _get_axis
from qpwcnet.core.agc import adaptive_clip_grad
from qpwcnet.core.vis import flow_to_image

from qpwcnet.train.loss import (
    FlowMseLoss,
    FlowMseLossFineTune,
    FlowMseLossV2,
    AdaptiveResizedLossV2,

    # NOTE(ycho): Inherited import..
    AdaptiveLossFunction
)

from qpwcnet.data.fchairs3d import get_dataset_from_set
from qpwcnet.data.tfrecord import get_reader, read_record
from qpwcnet.data.augment import image_augment, image_resize


def learning_rate(batch_size):
    base_lr = 0.0001
    # Fixed lr boundaries, by number of training samples (not number of
    # batches).
    lr_boundaries = [400000 * 8, 600000 * 8, 800000 * 8, 1000000 * 8]
    # Adjust the boundaries by batch size.
    lr_boundaries = [int(x / batch_size) for x in lr_boundaries]

    lr_values = [base_lr / (2**i) for i in range(len(lr_boundaries) + 1)]
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_boundaries, values=lr_values)
    return lr_scheduler


def learning_rate_cyclic(batch_size):
    # Fixed lr boundaries, by number of training samples (not number of
    # batches).
    lr = tfa.optimizers.Triangular2CyclicalLearningRate(
        initial_learning_rate=1e-4,
        maximal_learning_rate=5e-3,
        step_size=10e3 * (8 / batch_size)
    )
    return lr


def preprocess_no_op(ims, flo, data_format='channels_first'):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0 / 255.0, dtype=tf.float32)

    # resize, no augmentation.
    ims, flo = image_resize(ims, flo, (256, 512))

    # 0.0-1.0 -> -0.5, 0.5
    ims = ims - 0.5

    # HWC -> CHW
    if data_format == 'channels_first':
        ims = tf.transpose(ims, (2, 0, 1))
        flo = tf.transpose(flo, (2, 0, 1))
    return ims, flo


def preprocess(ims, flo, data_format='channels_first', base_scale=1.0):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0 / 255.0, dtype=tf.float32)
    # apply augmentation
    ims, flo = image_augment(ims, flo, (256, 512), base_scale)
    # 0.0-1.0 -> -0.5, 0.5
    ims = ims - 0.5

    # HWC -> CHW
    if data_format == 'channels_first':
        ims = tf.transpose(ims, (2, 0, 1))
        flo = tf.transpose(flo, (2, 0, 1))

    # NOTE(ycho): Uncomment to test NaN detection
    # flo += tf.constant(np.nan)

    # NOTE(ycho): Here, we're silently accepting NaNs
    # which apparently does exist in FlyingChairs3D.
    # A slightly better approach would be to filter them out
    # from the training set.
    ims = tf.where(tf.math.is_nan(ims), tf.zeros_like(ims), ims)
    flo = tf.where(tf.math.is_nan(flo), tf.zeros_like(flo), flo)

    return ims, flo


@tf.function
def train_step(model, losses, optim, ims, flo):
    # print('model.losses', model.losses)
    with tf.GradientTape() as tape:
        # Compute flow predictions.
        pred_flows = model(ims)

        # Compute loss over all flow pairs at each scale.
        flow_losses = [l(flo, o) for o, l in zip(pred_flows[:-1], losses)]

        # Finalize error with regularization/auxiliary losses
        loss = sum(flow_losses) + sum(model.losses)

    # Protection against NaN gradients.
    # Maybe this occurs due to the dataset?
    # tf.debugging.check_numerics(
    #    loss, "nan-loss"
    # )
    grads = tape.gradient(loss, model.trainable_variables)
    # for g, v in zip(grads, model.trainable_variables):
    #    tf.debugging.check_numerics(
    #        g, "nan-grad@" + v.name
    #    )
    grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) for g in grads]
    optim.apply_gradients(zip(grads, model.trainable_variables))
    return optim.iterations, flow_losses, loss


def setup_input(batch_size, data_format):

    # Load MPI Sintel dataset.
    # def _preprocess(ims, flo):
    #     return preprocess(ims, flo, data_format, 1.0)
    # glob_pattern = '/media/ssd/datasets/sintel-processed/shards/sintel-*.tfrecord'
    # dataset = (tf.data.Dataset.list_files(glob_pattern).interleave(
    #     lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'),
    #     cycle_length=tf.data.experimental.AUTOTUNE,
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #     .shuffle(buffer_size=32)
    #     .map(read_record)
    #     .map(_preprocess)
    #     .batch(batch_size, drop_remainder=True)
    #     .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

    # Load FlyingChairs3D dataset.
    # FIXME(yycho0108): 0.56 value here is an inevitable result of:
    # - the desire to downsample the image resolution
    # - the requirement that the size of the output image be 256x512.
    def _preprocess_fc3d(ims, flo):
        return preprocess(ims, flo, data_format, 0.56)
    dataset = (
        get_dataset_from_set()
        .map(_preprocess_fc3d,
             num_parallel_calls=tf.data.experimental.AUTOTUNE,
             deterministic=False)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    # dataset = dataset.concatenate(dataset_fc3d)
    # dataset = dataset.take(1).cache()
    return dataset


def setup_path(root='/tmp/pwc'):
    root = Path('/tmp/pwc')
    run_root = root / 'run'
    run_root.mkdir(parents=True, exist_ok=True)
    # NOTE(yycho0108): Automatically computing run id.
    run_id = len(list(run_root.iterdir()))
    run_dir = run_root / '{:03d}'.format(run_id)
    log_dir = run_dir / 'log'
    ckpt_dir = run_dir / 'ckpt'

    # Ensure directories exist.
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = {
        'pwc_root': root,
        'run_root': run_root,
        'run': run_dir,
        'ckpt': ckpt_dir,
        'log': log_dir,
        'id': run_id
    }
    return path


class ShowImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size: int,
                 log_dir: str, log_period: int = 128):
        self.batch_size = batch_size
        self.log_dir = Path(log_dir) / 'image'
        self.log_period = log_period

        self.batch_index = 0
        self.writer = tf.summary.create_file_writer(str(self.log_dir))
        self.val_data, self.val_flow_img = self._get_test_data()

    def _get_test_data(self):
        batch_size = self.batch_size
        data_format = tf.keras.backend.image_data_format()
        val_data = next(get_dataset_from_set().map(preprocess_no_op).batch(
            batch_size).take(1).cache().as_numpy_iterator())

        # Might as well also precompute flow image.
        val_ims, val_flo = val_data
        val_flow_img = flow_to_image(val_flo, data_format=data_format)
        if data_format == 'channels_first':
            # nchw -> nhwc
            val_flow_img = tf.transpose(val_flow_img, (0, 2, 3, 1))
        return val_data, val_flow_img

    def on_batch_end(self, batch, logs={}):
        self.batch_index += 1
        if (self.batch_index % self.log_period) != 0:
            return

        data_format = tf.keras.backend.image_data_format()

        val_ims, val_flo = self.val_data
        val_flow_img = self.val_flow_img

        flow_imgs = [val_flow_img]
        flows = self.model.predict(val_ims)
        for flow in flows:
            flow_img = flow_to_image(flow, data_format=data_format)
            if data_format == 'channels_first':
                # nchw -> nhwc
                flow_img = tf.transpose(flow_img, (0, 2, 3, 1))

            # NOTE(yycho0108):
            # interpolate nearest (tensorboard visualization applies
            # bilinear interpolation by default).
            flow_img = tf.image.resize(
                flow_img, size=val_flow_img.shape[1: 3],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            flow_imgs.append(flow_img)

        with self.writer.as_default():
            # will this work?
            for i, flow_img in enumerate(flow_imgs):
                name = 'flow-{:02d}'.format(i)
                tf.summary.image(name, flow_img, step=batch,
                                 max_outputs=3)


def epe_error(data_format='channels_last'):
    axis = -1 if (data_format == 'channels_last') else 1

    def _epe_error(y_true, y_pred):
        err = tf.norm(y_true - y_pred, ord=2, axis=axis)
        return tf.reduce_mean(err)
    return _epe_error


class TrainModel(tf.keras.Model):
    def __init__(self,
                 model: tf.keras.Model,
                 clip_factor: float = 0.01, eps: float = 1e-3):
        super().__init__(model.inputs, model.outputs)
        self.model = model
        self.clip_factor = clip_factor
        self.eps = eps
        data_format = tf.keras.backend.image_data_format()
        self.axis = _get_axis(data_format)

    def train_step(self, data):
        ims, flo = data
        with tf.GradientTape() as tape:
            pred_flows = self.model(ims, training=True)
            # NOTE(ycho): Exclude last one (full res image = untrainable)
            loss = self.compiled_loss(
                [flo for _ in pred_flows[:-1]],
                pred_flows[:-1],
                regularization_losses=self.losses)

        # OK, hopefully this has ARL variables now
        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)

        # NO nan grads will be allowed
        # TODO(ycho): Below code is probably inefficient and hacky.
        # It would be ideal to actually figure out why these nans are
        # happening.
        #has_nan = tf.reduce_any(
        #    [tf.reduce_any(tf.math.is_nan(g)) for g in grads])
        #def _apply_gradients():
        #    agc_grads = adaptive_clip_grad(
        #        params, grads, self.clip_factor, self.eps)
        #    self.optimizer.apply_gradients(zip(agc_grads, params))
        #    return 0
        #zero = tf.cond(has_nan, lambda: 0, _apply_gradients)

        agc_grads = adaptive_clip_grad(
            params, grads, self.clip_factor, self.eps)
        self.optimizer.apply_gradients(zip(agc_grads, params))

        # AGC == freedom from batchnorm?
        self.compiled_metrics.update_state(flo, pred_flows[-1])
        return {m.name: m.result() for m in self.metrics}

    def save_weights(self, *args, **kwargs):
        return self.model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.model.load_weights(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs, *args, **kwargs)


def train_keras(model, losses, dataset, path, config):

    # Unroll config.
    (batch_size, num_epoch, update_freq, data_format,
     allow_memory_growth, use_custom_training) = config

    callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(path['ckpt'] / '{epoch:03d}.ckpt')),
        tf.keras.callbacks.TensorBoard(
            update_freq=update_freq,  # every 128 batches
            log_dir=path['log'], profile_batch='2,66'),
        ShowImageCallback(batch_size, log_dir=path['log'] / 'flow',
                          log_period=update_freq)
    ]

    if False:
        # Rewrite model with ARL.
        # flow_input = tf.keras.Input(
        #     shape=(2,) + (256, 512),
        #     dtype=tf.float32, name='flow_label')

        # NOTE(ycho): Exclude last one (full res image = untrainable)
        alfs = [AdaptiveLossFunction(2, tf.float32, name='alf_{}'.format(i))
                for (i, o) in enumerate(model.outputs[:-1])]

        # Passthrough on outputs, but add loss along the way.
        losses = [AdaptiveResizedLossV2(alf) for alf in alfs]
        #outputs = [l(flow_input, o)
        #           for l, o in zip(losses, model.outputs)]

        # model = tf.keras.Model(model.inputs + [flow_input], outputs)

        model = TrainModel(model, alfs)
    else:
        model = TrainModel(model)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    optimizer = tf.keras.optimizers.Adam(
        # learning_rate=learning_rate_cyclic(batch_size)
        learning_rate=1.0e-4
    )

    # Restore from previous checkpoint.
    if True:
        # model.load_weights('/tmp/pwc/run/075/model.h5', by_name=True)
        model.load_weights('/tmp/pwc/run/119/model.h5', by_name=True)

        #latest_ckpt = tf.train.latest_checkpoint('/tmp/pwc/run/003/ckpt/')
        #tf.train.Checkpoint(optimizer=optimizer,
        #                    model=model).restore(latest_ckpt)
        ## or ...
        ## model.load_weights(latest_ckpt)?

    model.compile(
        optimizer=optimizer,
        loss=losses,
        # FIXME(yycho0108): This is super ugly, but probably works for now.
        metrics={'upsample_4': epe_error(data_format)}
    )

    try:
        model.fit(dataset,
                  epochs=num_epoch,
                  callbacks=callbacks)
    except KeyboardInterrupt as e:
        pass
    finally:
        for ext, fmt in zip(['pb', 'h5'], ['tf', 'h5']):
            out_file = str(path['run'] / 'model.{}'.format(ext))
            logging.info(
                'saving weights to {} prior to termination ...'.format(
                    out_file))
            model.save_weights(out_file, save_format=fmt)


def train_custom(model, losses, dataset, path, config):
    """
    Custom training loop.
    """

    # Unroll config.
    (batch_size, num_epoch, update_freq, data_format,
     allow_memory_growth, use_custom_training) = config

    # Setup metrics.
    metrics = {}
    metrics['loss'] = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
    # metrics['epe'] = tf.keras.metrics.Mean(name='epe', dtype=tf.float32)
    for out in model.outputs:
        if data_format == 'channels_first':
            h = out.shape[2]
        else:
            h = out.shape[1]
        name = 'flow-loss-{:02d}'.format(h)
        metrics[name] = tf.keras.metrics.Mean(name=name, dtype=tf.float32)

    # Retrieve validation dataset (only used for visualization for now) ...
    val_data = next(get_dataset_from_set().map(preprocess_no_op).batch(
        batch_size).take(1).cache().as_numpy_iterator())

    # Setup handlers for training/logging.
    # lr = learning_rate_cyclic(batch_size)
    lr = 1e-4  # learning_rate_cyclic(batch_size)
    optim = tf.keras.optimizers.Adam(learning_rate=lr)
    writer = tf.summary.create_file_writer(str(path['log']))
    ckpt = tf.train.Checkpoint(optimizer=optim, model=model)
    ckpt_mgr = tf.train.CheckpointManager(
        ckpt, str(path['ckpt']), max_to_keep=8)

    # Load from checkpoint.
    ckpt.restore(tf.train.latest_checkpoint('/tmp/pwc/run/044/ckpt/'))

    # Iterate through train loop.
    for epoch in range(num_epoch):
        print('Epoch {:03d}/{:03d}'.format(epoch, num_epoch))
        # prepare epoch.
        for v in metrics.values():
            v.reset_states()

        # train epoch.
        for ims, flo in dataset:
            # Skip invalid inputs (unlikely but happens sometimes)
            if not (tf.reduce_all(tf.math.is_finite(ims))
                    and tf.reduce_all(tf.math.is_finite(flo))):
                continue

            opt_iter, flow_loss, step_loss = train_step(
                model, losses, optim, ims, flo)

            # update metrics.
            metrics['loss'].update_state(step_loss)
            for out, l in zip(model.outputs, flow_loss):
                if data_format == 'channels_first':
                    h = out.shape[2]
                else:
                    h = out.shape[1]
                name = 'flow-loss-{:02d}'.format(h)
                metrics[name].update_state(l)

            # log/save.
            if (opt_iter > 0) and ((opt_iter % update_freq) == 0):
                # compute flows and output image.
                val_ims, val_flo = val_data

                # First add ground truth flow ...
                val_flow_img = flow_to_image(val_flo, data_format=data_format)
                if data_format == 'channels_first':
                    # nchw -> nhwc
                    val_flow_img = tf.transpose(val_flow_img, (0, 2, 3, 1))
                flow_imgs = [val_flow_img]

                flows = model(val_ims, training=False)
                for flow in flows:
                    flow_img = flow_to_image(flow, data_format=data_format)
                    if data_format == 'channels_first':
                        # nchw -> nhwc
                        flow_img = tf.transpose(flow_img, (0, 2, 3, 1))

                    # NOTE(yycho0108):
                    # interpolate nearest (tensorboard visualization applies
                    # bilinear interpolation by default).
                    flow_img = tf.image.resize(
                        flow_img, size=val_flow_img.shape[1: 3],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    flow_imgs.append(flow_img)

                with writer.as_default():
                    tf.summary.scalar('iter', opt_iter, step=opt_iter)
                    tf.summary.scalar('learning_rate', lr, step=opt_iter)
                    # tf.summary.scalar('learning_rate', lr(
                    #    tf.cast(opt_iter, tf.float32)), step=opt_iter)
                    for k, v in metrics.items():
                        tf.summary.scalar(k, v.result(), step=opt_iter)
                    # will this work?
                    for i, flow_img in enumerate(flow_imgs):
                        name = 'flow-{:02d}'.format(i)
                        tf.summary.image(name, flow_img, step=opt_iter,
                                         max_outputs=3)
        ckpt_mgr.save(epoch)
    model.save_weights(str(path['run'] / 'model.h5'))


def main():
    logging.basicConfig(level='INFO')

    # Configure hyperparameters.
    batch_size = 16
    num_epoch = 600
    update_freq = 16
    data_format = 'channels_first'
    allow_memory_growth = False
    use_custom_training = False

    tf.keras.backend.set_image_data_format(data_format)

    # TODO(yycho0108): Better configuration management.
    config = (batch_size, num_epoch, update_freq, data_format,
              allow_memory_growth, use_custom_training)

    # Configure memory growth.
    if allow_memory_growth:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Setup directory structure.
    path = setup_path()
    print('Run id = {}'.format(path['id']))

    if False:
        tf.debugging.enable_check_numerics()

    # if True:
    #    tf.debugging.experimental.enable_dump_debug_info(
    #        logdir=path['run'] / 'debug'"/tmp/tfdbg2_logdir",
    #        tensor_debug_mode="FULL_HEALTH",
    #        circular_buffer_size=-1)

    dataset = setup_input(batch_size, data_format)

    model = build_network(train=True, data_format=data_format)

    # Create losses, flow mse
    # NOTE(yycho0108): The final output (result of bilinear interpolation) is
    # not included.
    # NOTE(ycho): Exclude last one (full res image = untrainable)
    losses = [FlowMseLossV2() for o in model.outputs[:-1]]
    # losses = [FlowMseLossFineTune(data_format=data_format)
    #          for o in model.outputs[:-1]]
    # losses = [AdaptiveResizedLossV2(o.shape) for o in model.outputs[:-1]]
    # losses = None

    # Train.
    with open(path['run'] / 'config.txt', 'w') as f_cfg:
        names = ['batch_size', 'num_epoch', 'update_freq',
                 'data_format', 'allow_memory_growth', 'use_custom_training']
        values = config
        cfg = {k: v for (k, v) in zip(names, values)}
        f_cfg.write('{}'.format(cfg))

    if use_custom_training:
        train_custom(model, losses, dataset, path, config)
    else:
        train_keras(model, losses, dataset, path, config)


if __name__ == '__main__':
    main()
