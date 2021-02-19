#!/usr/bin/env python3

import tensorflow as tf
from dataclasses import dataclass
from simple_parsing import Serializable
from qpwcnet.app.arg_setup import with_args
from typing import Tuple
from pathlib import Path
import json

from qpwcnet.data.youtube_vos import YoutubeVos, YoutubeVosSettings, triplet_dataset
from qpwcnet.core.pwcnet import build_interpolator
from qpwcnet.core.layers import _get_axis
from qpwcnet.core.agc import adaptive_clip_grad
from qpwcnet.train.loss import AutoResizeMseLoss


@dataclass
class Settings(Serializable):
    root: str = '/tmp/pwc'
    batch_size: int = 8
    num_epoch: int = 600
    update_freq: int = 16
    data_format: str = 'channels_first'
    allow_memory_growth: bool = True
    debug_nan: bool = False
    learning_rate: float = 2.5e-4
    input_shape: Tuple[int, int] = (256, 512)


class TrainAgcModel(tf.keras.Model):
    def __init__(self, model: tf.keras.Model,
                 clip_factor: float = 0.01, eps: float = 1e-3):
        super().__init__(model.inputs, model.outputs)
        self.model = model
        self.clip_factor = clip_factor
        self.eps = eps
        data_format = tf.keras.backend.image_data_format()
        self.axis = _get_axis(data_format)

    def train_step(self, data):
        img0, img1, img2 = data
        img_pair = tf.concat([img0, img2], axis=self.axis)
        with tf.GradientTape() as tape:
            pred_imgs = self.model(img_pair)
            loss = self.compiled_loss(
                [img1 for _ in pred_imgs],  # label
                pred_imgs,  # prediction
                regularization_losses=self.losses)

        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)

        # AGC == freedom from batchnorm?
        agc_grads = adaptive_clip_grad(
            params, grads, self.clip_factor, self.eps)

        self.optimizer.apply_gradients(zip(agc_grads, params))
        self.compiled_metrics.update_state(img1, pred_imgs[-1])
        return {m.name: m.result() for m in self.metrics}

    def save_weights(self, *args, **kwargs):
        self.model.save_weights(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs, *args, **kwargs)


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


def train(args: Settings, model: tf.keras.Model,
          dataset: tf.data.Dataset, path):
    callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(path['ckpt'] / '{epoch:03d}.hdf5')),
        tf.keras.callbacks.TensorBoard(
            update_freq=args.update_freq,  # every 128 batches
            log_dir=path['log'], profile_batch='2,66'),
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    losses = [AutoResizeMseLoss() for _ in model.outputs]
    model.compile(
        optimizer=optimizer,
        loss=losses,
        # metrics=[tf.keras.metrics.MeanSquaredError()]
    )

    model.fit(dataset,
              epochs=args.num_epoch,
              callbacks=callbacks)
    model.save_weights(str(path['run'] / 'model.h5'))


@with_args(Settings)
def main(args):

    # Configure memory growth.
    if args.allow_memory_growth:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Setup directory structure.
    path = setup_path(args.root)
    print('Run id = {}'.format(path['id']))

    if args.debug_nan:
        tf.debugging.enable_check_numerics()

    dataset = YoutubeVos(YoutubeVosSettings(data_type='train'))
    dataset = triplet_dataset(
        dataset,
        dsize=args.input_shape,
        batch_size=args.batch_size)

    model = build_interpolator(args.input_shape)
    train_model = TrainAgcModel(model)

    # Save cfg
    with open(path['run'] / 'config.json', 'w') as f_cfg:
        args.dump(f_cfg)

    # Train.
    train(args, train_model, dataset, path)


if __name__ == '__main__':
    main()
