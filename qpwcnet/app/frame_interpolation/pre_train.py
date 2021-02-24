#!/usr/bin/env python3

from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Tuple
from pathlib import Path
import json
import logging

import tensorflow as tf
import tensorflow_addons as tfa
import einops

from qpwcnet.core.pwcnet import build_interpolator
from qpwcnet.core.layers import _get_axis
from qpwcnet.core.agc import adaptive_clip_grad
from qpwcnet.train.loss import AutoResizeMseLoss

from qpwcnet.data.youtube_vos import (YoutubeVosTriplet, YoutubeVosSettings,
                                      YoutubeVosTripletSettings)
from qpwcnet.data.vimeo_triplet import (
    VimeoTriplet, VimeoTripletSettings)
from qpwcnet.data.triplet_dataset_ops import read_triplet_dataset

from qpwcnet.app.util.arg_setup import with_args


@dataclass
class Settings(Serializable):
    root: str = '/tmp/pwc'
    batch_size: int = 8
    num_epoch: int = 600
    update_freq: int = 256
    data_format: str = 'channels_first'
    allow_memory_growth: bool = False
    debug_nan: bool = False
    learning_rate: float = 1.0e-4
    input_shape: Tuple[int, int] = (256, 512)
    load_ckpt: str = ''
    dataset: str = 'vimeo'
    log_level: str = 'INFO'


class TrainModel(tf.keras.Model):
    def __init__(self, model: tf.keras.Model,
                 clip_factor: float = 0.01, eps: float = 1e-3):
        super().__init__(model.inputs, model.outputs)
        self.model = model
        self.clip_factor = clip_factor
        self.eps = eps
        data_format = tf.keras.backend.image_data_format()
        self.axis = _get_axis(data_format)

    def train_step(self, data):
        img_pair, img1 = data
        with tf.GradientTape() as tape:
            pred_imgs = self.model(img_pair, training=True)
            loss = self.compiled_loss(
                [img1 for _ in pred_imgs],  # label(s)
                pred_imgs,  # prediction(s)
                regularization_losses=self.losses)

        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)

        # AGC == freedom from batchnorm?
        agc_grads = adaptive_clip_grad(
            params, grads, self.clip_factor, self.eps)
        self.optimizer.apply_gradients(zip(agc_grads, params))
        # self.optimizer.apply_gradients(zip(grads, params))
        self.compiled_metrics.update_state(img1, pred_imgs[-1])
        return {m.name: m.result() for m in self.metrics}

    def save_weights(self, *args, **kwargs):
        return self.model.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.model.load_weights(*args, **kwargs)

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


def preprocess(img0, img1, img2):
    # Normalize to (-0.5, 0.5) + concat pairs
    img0 -= 0.5
    img1 -= 0.5
    img2 -= 0.5
    img_pair = tf.concat([img0, img2], axis=-1)

    # Deal with data format.
    data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_first':
        # img_pair = tf.transpose(img_pair, (0, 3, 1, 2))
        # img1 = tf.transpose(img1, (0, 3, 1, 2))
        img_pair = einops.rearrange(img_pair, '... h w c -> ... c h w')
        img1 = einops.rearrange(img1, '... h w c -> ... c h w')
    return (img_pair, img1)


class ShowImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, cfg: Settings,
                 log_dir: str, log_period: int = 128):
        self.cfg = cfg
        self.log_dir = Path(log_dir) / 'image'
        self.log_period = log_period

        self.batch_index = 0
        self.writer = tf.summary.create_file_writer(str(self.log_dir))
        self.imgs = self._get_test_triplet()
        self.inputs = preprocess(*self.imgs)

    def _get_test_triplet(self):
        dataset = VimeoTriplet(VimeoTripletSettings(data_type='test'))
        dataset = read_triplet_dataset(dataset,
                                       dsize=self.cfg.input_shape,
                                       shuffle=True,
                                       augment=False,
                                       prefetch=False,
                                       batch_size=1)
        for out in dataset:
            break
        # Explicitly delete dataset,
        # to avoid large memory consumption.
        del dataset
        return out

    def on_batch_end(self, batch, logs={}):
        self.batch_index += 1
        if (self.batch_index % self.log_period) != 0:
            return

        pred_imgs = self.model.predict(self.inputs[0])
        pred_img1 = pred_imgs[-1]

        data_format = tf.keras.backend.image_data_format()
        if data_format == 'channels_first':
            pred_img1 = einops.rearrange(pred_img1, '... c h w -> ... h w c')

        overlay = 0.5 * self.imgs[0] + 0.5 * self.imgs[2]
        with self.writer.as_default():
            # NOTE(ycho): [None, ...] to add batch dimension:
            # Apparently image summary needs to be rank 4.
            tf.summary.image('img0', self.imgs[0], step=batch)
            tf.summary.image('img1', self.imgs[1], step=batch)
            tf.summary.image('img2', self.imgs[2], step=batch)
            tf.summary.image('overlay', overlay, step=batch)
            # NOTE(ycho): +0.5 to undo preprocess()
            tf.summary.image('pred-img1', 0.5 + pred_img1, step=batch)


def train(args: Settings, model: tf.keras.Model,
          dataset: tf.data.Dataset, path):
    callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(path['ckpt'] / '{epoch:03d}.ckpt')),
        tf.keras.callbacks.TensorBoard(
            update_freq=args.update_freq,  # every 128 batches
            log_dir=path['log'], profile_batch='2,66'),
        ShowImageCallback(args, log_dir=path['log'] / 'flow',
                          log_period=args.update_freq)
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    losses = [AutoResizeMseLoss() for _ in model.outputs]
    model.compile(
        optimizer=optimizer,
        loss=losses,
        # metrics=[tf.keras.metrics.MeanSquaredError()]
    )

    if args.load_ckpt:
        weight_path = Path(args.load_ckpt) / 'variables/variables'
        model.load_weights(weight_path,
                           by_name=False)

    try:
        model.fit(dataset,
                  epochs=args.num_epoch,
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


@with_args(Settings)
def main(args):
    logging.basicConfig(level=args.log_level)
    # global data format setting
    tf.keras.backend.set_image_data_format(args.data_format)

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

    # Select dataset.
    if args.dataset == 'ytvos':
        dataset = YoutubeVos(YoutubeVosSettings(data_type='train'))
    elif args.dataset == 'vimeo':
        dataset = VimeoTriplet(VimeoTripletSettings(data_type='train'))
    else:
        raise ValueError('Invalid dataset = {}'.format(args.dataset))

    # TripletDataset -> tf.data.Dataset
    dataset = read_triplet_dataset(dataset, dsize=args.input_shape,
                                   batch_size=args.batch_size)

    # Preprocess = Normalize + transpose
    dataset = dataset.map(preprocess)

    model = build_interpolator(args.input_shape)

    train_model = TrainModel(model)

    # Save cfg
    with open(path['run'] / 'config.json', 'w') as f_cfg:
        args.dump(f_cfg)

    # Train.
    train(args, train_model, dataset, path)


if __name__ == '__main__':
    main()
