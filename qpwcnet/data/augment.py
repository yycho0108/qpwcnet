#!/usr/bin/env python3

import tensorflow as tf
import einops

from tensorflow.python.framework import tensor_shape
from typing import Tuple


def rotation_matrix_from_euler(x: tf.Tensor):
    c, s = tf.cos(x), tf.sin(x)
    cx, cy, cz = tf.unstack(c, axis=-1)
    sx, sy, sz = tf.unstack(s, axis=-1)

    R = [cy * cz,
         (sx * sy * cz) - (cx * sz),
         (cx * sy * cz) + (sx * sz),
         cy * sz,
         (sx * sy * sz) + (cx * cz),
         (cx * sy * sz) - (sx * cz),
         -sy,
         sx * cy,
         cx * cy]
    R = einops.rearrange(R, '(b c) ... -> ... b c', b=3, c=3)
    return R


def restore_shape(t0: tf.Tensor, t1: tf.Tensor):
    shape = t0.get_shape()
    if shape == tensor_shape.unknown_shape():
        t1.set_shape([None, None, None])
    else:
        t1.set_shape(shape)
    return t1


def photometric_augmentation(
        image: tf.Tensor, z_shape: Tuple[int],
        max_txn: float = 0.3,
        max_rxn: float = 0.3,
        max_scale: float = 0.3):
    """
    NOTE(ycho): max_scale is specified in log space.
    NOTE(ycho): len(z_shape) == rank(image)
    """

    # Generate random variables.
    z_txn = tf.random.uniform(z_shape + (3,), minval=-max_txn, maxval=+max_txn)
    z_rxn = tf.random.uniform(z_shape + (3,), minval=-max_rxn, maxval=+max_rxn)
    z_scale = tf.exp(tf.random.uniform(
        z_shape + (3,),
        minval=-max_scale, maxval=+max_scale))

    # Convert + apply
    R = rotation_matrix_from_euler(z_rxn)
    x = tf.einsum('...ab,...b->...a', R, image)
    out = x * z_scale + z_txn
    out = restore_shape(image, out)
    return out


def image_augment_colors(ims):
    # Generate random variables.
    hue_delta = tf.random.uniform([], minval=-0.2, maxval=0.2)
    brightness_delta = tf.random.uniform([], minval=-0.125, maxval=0.125)
    saturation_delta = tf.random.uniform([], minval=0.5, maxval=1.5)
    contrast_delta = tf.random.uniform([], minval=0.5, maxval=1.5)

    # to batch
    ims = einops.rearrange(ims, 'h w (k c) -> h (w k) c', k=2)

    # augment
    ims = tf.image.adjust_brightness(ims, brightness_delta)
    ims = tf.image.adjust_saturation(ims, saturation_delta)
    ims = tf.image.adjust_hue(ims, hue_delta)
    ims = tf.image.adjust_contrast(ims, contrast_delta)

    # to depth
    ims = einops.rearrange(ims, 'h (w k) c -> h w (k c)', k=2)
    return ims


def image_flip_ud(ims, flo):
    distort_up_down_random = tf.random.uniform(
        [], 0, 1.0, dtype=tf.float32)
    flip = tf.less(distort_up_down_random, 0.5)
    flip_mask = tf.stack([flip, False, False])
    flip_axis = tf.boolean_mask([0, 1, 2], flip_mask)
    sign = 1.0 - 2 * tf.cast(flip, tf.float32)  # 1->-1, 0->1

    ims = tf.reverse(ims, flip_axis)
    flo = tf.reverse(flo, flip_axis)
    u = flo[:, :, :1]
    v = flo[:, :, 1:] * sign
    flo = tf.concat([u, v], axis=2)

    return ims, flo


def image_flip_lr(ims, flo):
    distort_left_right_random = tf.random.uniform(
        [], 0, 1.0, dtype=tf.float32)
    flip = tf.less(distort_left_right_random, 0.5)
    flip_mask = tf.stack([False, flip, False])
    flip_axis = tf.boolean_mask([0, 1, 2], flip_mask)
    sign = 1.0 - 2 * tf.cast(flip, tf.float32)  # 1->-1, 0->1

    ims = tf.reverse(ims, flip_axis)
    flo = tf.reverse(flo, flip_axis)
    u = flo[:, :, :1] * sign
    v = flo[:, :, 1:]
    flo = tf.concat([u, v], axis=2)

    return ims, flo


def image_scale_and_crop(ims, flo, crop_shape, base_scale=1.0):
    im_concat = tf.concat([ims, flo], axis=2)
    # TODO(yycho0108): Fix hardcoded 0.955/1.05.
    scale = tf.random.uniform(
        [],
        minval=base_scale * 0.955,
        maxval=base_scale * 1.05,
        dtype=tf.float32,
        seed=None)
    # scaled_shape = tf.cast(
    #    tf.cast(ims.shape[:2], tf.float32) * scale, tf.int32)

    ims_shape = tf.shape(ims)
    scaled_shape = tf.cast(
        tf.cast(ims_shape[:2], tf.float32) * scale, tf.int32)

    im_resized = tf.image.resize(
        im_concat, scaled_shape, method=tf.image.ResizeMethod.BILINEAR)
    im_cropped = tf.image.random_crop(
        im_resized, [crop_shape[0], crop_shape[1], 8])

    ims = im_cropped[:, :, :6]  # :-2? to be robust for monochrome
    flo = im_cropped[:, :, 6:]
    flo = flo * scale

    return ims, flo


def image_resize(ims, flo, shape):
    scale = tf.truediv(shape, tf.shape(ims)[:2])
    im_concat = tf.concat([ims, flo], axis=2)
    im_resized = tf.image.resize(
        im_concat, shape, method=tf.image.ResizeMethod.BILINEAR)
    ims = im_resized[:, :, :6]
    flo = im_resized[:, :, 6:]
    flo = flo * [scale[1], scale[0]]
    return ims, flo


def image_crop(ims, flo, crop_shape):
    im_concat = tf.concat([ims, flo], axis=2)
    im_cropped = tf.image.random_crop(
        im_concat, [crop_shape[0],
                    crop_shape[1],
                    8])  # RGB + RGB + UV = 8 channels
    ims = im_cropped[:, :, :6]
    flo = im_cropped[:, :, 6:]
    return ims, flo


def image_augment(ims, flo, out_shape, base_scale=1.0):
    ims, flo = image_flip_ud(ims, flo)
    ims, flo = image_flip_lr(ims, flo)
    ims, flo = image_scale_and_crop(ims, flo, out_shape, base_scale=base_scale)
    ims, flo = image_resize(ims, flo, out_shape)
    ims = image_augment_colors(ims)
    return ims, flo
