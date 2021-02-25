#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import einops
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from typing import Tuple
from functools import partial

from qpwcnet.core.pwcnet import build_flower
from qpwcnet.core.warp import tf_warp
from qpwcnet.core.vis import flow_to_image
from qpwcnet.vis.show import show
from qpwcnet.train.util import load_weights

from qpwcnet.data.tfrecord import get_reader
from qpwcnet.data.augment import image_resize, image_augment

from qpwcnet.app.arg_setup import with_args


def normalize(x):
    x = np.asarray(x)
    minx, maxx = x.min(), x.max()
    return (x - minx) / (maxx - minx)


def preprocess(ims, flo):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0 / 255.0, dtype=tf.float32)
    # resize, no augmentation.
    # ims, flo = image_resize(ims, flo, (256, 512))
    ims, flo = image_augment(ims, flo, (256, 512))
    # 0.0-1.0 -> -0.5, 0.5
    ims = ims - 0.5

    # Convert to correct data format
    data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_first':
        ims = einops.rearrange(ims, '... h w c -> ... c h w')
        flo = einops.rearrange(flo, '... h w c -> ... c h w')

    return ims, flo


@dataclass
class Settings:
    model_file: str
    input_shape: Tuple[int, int]
    data_format: str = 'channels_first'


@with_args(Settings)
def main(cfg: Settings):
    if cfg.data_format is not None:
        tf.keras.backend.set_image_data_format(cfg.data_format)
    data_format = tf.keras.backend.image_data_format()

    # 1) Build inference-only network
    model = build_flower(False,
                         cfg.input_shape,
                         data_format)

    # 2) Restore model.
    load_weights(model, cfg.model_file)

    if False:
        # from image

        # x = np.random.uniform(0, 255, size=(1, 256, 512, 6)).astype(np.uint8)
        lhs = cv2.imread(
            '/media/ssd/datasets/MPI-Sintel-complete/test/final/ambush_3/frame_0014.png')
        rhs = cv2.imread(
            '/media/ssd/datasets/MPI-Sintel-complete/test/final/ambush_3/frame_0015.png')
        lhs = cv2.resize(lhs, (512, 256))
        rhs = cv2.resize(rhs, (512, 256))
        x = np.concatenate([lhs, rhs], axis=-1)[None, ...]
        # FIXME(yycho0108): the series of above operations replicate
        # preprocess() data whitening procedure.
        y = model(x / 255.0 - 0.5).numpy()
        rhs_w = tf_warp(rhs[None, ...].astype(np.float32) / 255.0,
                        y)[0].numpy()

        cv2.imshow('lhs', lhs)
        cv2.imshow('rhs', rhs)
        cv2.imshow('overlay', rhs // 2 + lhs // 2)
        cv2.imshow('overlay-w', rhs_w / 2 + lhs / 255.0 / 2)
        cv2.imshow('flow-x', normalize(y[0, ..., 0]))
        cv2.imshow('flow-y', normalize(y[0, ..., 1]))
        cv2.imshow('rhs-w', rhs_w)
        cv2.waitKey(0)

    if True:
        # from tfrecord
        glob_pattern = '/media/ssd/datasets/sintel-processed/shards/sintel-*.tfrecord'
        filenames = tf.data.Dataset.list_files(glob_pattern).shuffle(32)
        # dataset = get_reader(filenames).shuffle(buffer_size=1024).repeat().batch(8)
        # dataset = get_reader(filenames).batch(8).repeat()
        dataset = get_reader(filenames).shuffle(
            buffer_size=32).map(preprocess).batch(1)

        for ims, flo in dataset:
            flo_pred = model.predict(ims)

            # Unstack `ims`.
            if data_format == 'channels_first':
                prv, nxt = einops.rearrange(
                    ims, 'n (k c) h w -> k n c h w', k=2)
            else:
                prv, nxt = einops.rearrange(
                    ims, 'n h w (k c) -> k n c h w', k=2)

            # NOTE(ycho): Maintain consistent `data_format` for sanity
            # preserving. Slightly inefficient but oh well...
            #if data_format == 'channels_first':
            #    nxt_nhwc = einops.rearrange(nxt, 'n c h w -> n h w c')
            #    flo_pred_nhwc = einops.rearrange(
            #        flo_pred, 'n c h w -> n h w c')
            #    nxt_w = tf_warp(nxt_nhwc, flo_pred_nhwc, data_format)
            #    nxt_w = einops.rearrange(nxt_w, 'n h w c -> n c h w')

            #    nxt_w_gt = tf_warp(nxt, flo)
            #else:
            nxt_w = tf_warp(nxt, flo_pred, data_format)
            nxt_w_gt = tf_warp(nxt, flo, data_format)

            # Undo `preprocess()`
            prv = 0.5 + prv
            nxt = 0.5 + nxt
            nxt_w = 0.5 + nxt_w
            nxt_w_gt = 0.5 + nxt_w_gt
            flo_pred = flo_pred

            # Apply colorization.
            flo_rgb = flow_to_image(flo, data_format)
            flo_pred_rgb = flow_to_image(flo_pred, data_format)

            # Compute derived visualizations.
            overlay = 0.5 * prv + 0.5 * nxt
            overlay_warped = 0.5 * prv + 0.5 * nxt_w
            delta_warped = tf.abs(0.5 * prv - 0.5 * nxt_w)
            overlay_warped_gt = 0.5 * prv + 0.5 * nxt_w_gt
            delta_warped_gt = tf.abs(0.5 * prv - 0.5 * nxt_w_gt)

            # Show all.
            for name in ['prv', 'nxt', 'nxt_w', 'overlay',
                         'overlay_warped', 'overlay_warped_gt',
                         'delta_warped', 'delta_warped_gt',
                         'flo_rgb', 'flo_pred_rgb']:
                image = locals()[name]
                # NOTE(ycho): unbatch before showing.
                show(name, image[0], True, data_format)

            k = cv2.waitKey(0)
            if k == 27:
                break


if __name__ == '__main__':
    main()
