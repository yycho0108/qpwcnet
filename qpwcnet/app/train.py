#!/usr/bin/env python3

from pathlib import Path
import tensorflow as tf
import numpy as np

from qpwcnet.core.pwcnet import build_network
from qpwcnet.data.fchairs3d import get_dataset, decode_files
from qpwcnet.data.tfrecord import get_reader, read_record
from qpwcnet.data.augment import image_augment, image_resize

from qpwcnet.train.loss import FlowMseLoss, FlowMseLossFineTune


def learning_rate(batch_size):
    base_lr = 0.0001
    # Fixed lr boundaries, by number of training samples (not number of batches).
    lr_boundaries = [400000*8, 600000*8, 800000*8, 1000000*8]
    # Adjust the boundaries by batch size.
    lr_boundaries = [int(x/batch_size) for x in lr_boundaries]

    lr_values = [base_lr/(2**i) for i in range(len(lr_boundaries)+1)]
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_boundaries, values=lr_values)
    return lr_scheduler


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
        flow_losses = [l(flo, o) for o, l in zip(pred_flows[:-1], losses[:-1])]

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


def epe_error(flo_label, flo_outputs):
    flo_pred = flo_outputs[-1]
    flo_gt = flo_label
    err = tf.norm(flo_pred - flo_gt, ord=2, axis=-1)
    return tf.reduce_mean(err)


def main():
    # Configure hyperparameters.
    batch_size = 4
    num_epoch = 600
    update_freq = 128

    # Configure memory growth.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

    # eager mode is just a recipe for uncompiled disasters...?
    # tf.compat.v1.disable_eager_execution()

    glob_pattern = '/media/ssd/datasets/sintel-processed/shards/sintel-*.tfrecord'
    filenames = tf.data.Dataset.list_files(glob_pattern).shuffle(32)
    # dataset = get_reader(filenames).shuffle(buffer_size=1024).repeat().batch(8)
    # dataset = get_reader(filenames).batch(8).repeat()

    # dataset = (get_reader(filenames)
    #           .shuffle(buffer_size=32)
    #           .map(preprocess)
    #           .batch(batch_size, drop_remainder=True)
    #           .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

    # sintel...
    dataset = tf.data.Dataset.list_files(glob_pattern).interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'),
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(buffer_size=32).map(read_record).map(preprocess).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # fchairs3d ...
    dataset_fc3d = get_dataset().shuffle(buffer_size=1024).map(
        decode_files).map(preprocess).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # merge ...
    dataset = dataset.concatenate(dataset_fc3d)

    # dataset = dataset.take(1).cache()
    model = build_network(train=True)

    # Create losses, flow mse
    losses = []
    for out in model.outputs[:-1]:
        losses.append(FlowMseLoss())
        # losses.append(FlowMseLossFineTune())

    # Setup directory structure.
    root = Path('/tmp/pwc')
    run_root = root / 'run'
    run_root.mkdir(parents=True, exist_ok=True)
    # NOTE(yycho0108): Automatically computing run id.
    run_id = len(list(run_root.iterdir()))
    run_dir = run_root / '{:03d}'.format(run_id)
    log_dir = run_dir / 'log'
    ckpt_dir = run_dir / 'ckpt'
    print('run_id = {}'.format(run_id))

    if False:
        # Custom training loop.

        # Setup metrics.
        metrics = {}
        metrics['loss'] = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
        for out in model.outputs:
            name = 'flow-loss-{:02d}'.format(out.shape[1])
            metrics[name] = tf.keras.metrics.Mean(name=name, dtype=tf.float32)

        # Setup handlers for training/logging.
        lr = learning_rate(batch_size)
        # lr = 6.25e-6
        optim = tf.keras.optimizers.Adam(learning_rate=lr)
        writer = tf.summary.create_file_writer(str(log_dir))
        ckpt = tf.train.Checkpoint(optimizer=optim, net=model)
        ckpt_mgr = tf.train.CheckpointManager(
            ckpt, str(ckpt_dir), max_to_keep=8)

        # [Optional] load checkpoint.
        if False:
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
                tf.summary.scalar('learning_rate', lr(opt_iter), step=epoch)
                # tf.summary.scalar('learning_rate', lr, step=epoch)
                for k, v in metrics.items():
                    tf.summary.scalar(k, v.result(), step=epoch)
            ckpt_mgr.save(epoch)
        model.save_weights(str(run_dir / 'model.h5'))

    if True:
        callbacks = [
            # tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_dir / '{epoch:03d}.hdf5')),
            tf.keras.callbacks.TensorBoard(
                update_freq=update_freq,  # every 128 batches
                log_dir=log_dir, profile_batch='2,66'),
        ]
        # The following perhaps "standard" keras training loop has been removed
        # due to a stupid memory leak(?)
        # Create losses, flow epe
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate(batch_size)),
            loss=losses,
            # FIXME(yycho0108): This is super ugly, but probably works for now.
            metrics={'upsample_4': epe_error}
        )
        #print('model losses')
        #print(model.losses)
        #print([t.name for t in model.losses])
        # return
        model.fit(dataset,
                  epochs=100,
                  callbacks=callbacks)
        model.save_weights(str(run_dir / 'model.h5'))


if __name__ == '__main__':
    main()
