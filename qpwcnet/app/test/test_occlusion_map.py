#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import faulthandler
import cv2
import einops

from qpwcnet.core.util import disable_gpu
from qpwcnet.core.occlusion import estimate_occlusion_map
from qpwcnet.core.warp import tf_warp
from qpwcnet.data.augment import image_resize, image_augment
from qpwcnet.data.tfrecord import get_reader
from qpwcnet.vis.show import show


def preprocess(ims, flo):
    # 0-255 -> 0.0-1.0
    ims = tf.cast(ims, tf.float32) * tf.constant(1.0 / 255.0, dtype=tf.float32)
    ims, flo = image_resize(ims, flo, (256, 512))
    ims = ims - 0.5

    # Convert to correct data format
    data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_first':
        ims = einops.rearrange(ims, '... h w c -> ... c h w')
        flo = einops.rearrange(flo, '... h w c -> ... c h w')

    return ims, flo


if True:
    tf.keras.backend.set_image_data_format('channels_last')
    data_format = tf.keras.backend.image_data_format()
    disable_gpu()

    # TODO(ycho): Cleanup dataset loading pattern for opt-flow datasets.
    glob_pattern = '/media/ssd/datasets/sintel-processed/shards/sintel-*.tfrecord'
    filenames = tf.data.Dataset.list_files(glob_pattern).shuffle(32)
    # dataset = get_reader(filenames).shuffle(buffer_size=1024).repeat().batch(8)
    # dataset = get_reader(filenames).batch(8).repeat()
    dataset = get_reader(filenames).shuffle(
        buffer_size=32).map(preprocess).batch(1)

    for ims, flo in dataset:
        # Unstack `ims`.
        if data_format == 'channels_first':
            prv, nxt = einops.rearrange(
                ims, 'n (k c) h w -> k n c h w', k=2)
        else:
            prv, nxt = einops.rearrange(
                ims, 'n h w (k c) -> k n h w c', k=2)

        occ = estimate_occlusion_map(flo, data_format)
        occ_f = tf.cast(occ, tf.float32)[..., None]
        nxt_w = tf_warp(nxt, flo, data_format)
        nxt_w_o = nxt_w * (1.0 - occ_f)

        show('occ', occ_f[0], True)
        show('prv', 0.5 + prv[0], True)
        show('nxt', 0.5 + nxt[0], True)
        show('nxt_w', 0.5 + nxt_w[0], True)
        show('nxt_w_o', ((0.5 + nxt_w_o) * (1.0 - occ_f))[0], True)
        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            break
