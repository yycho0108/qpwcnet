#!/usr/bin/env python3

import math
import tensorflow as tf

import einops


def cost_volume_to_flow(cvol: tf.Tensor, data_format: str = None):
    """
    Predict optical flow from cost volume by finding the argmax of correlation.
    """
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_last':
        axis = -1
        dims = einops.parse_shape(cvol, '... c')['c']
    else:
        axis = -3
        # dims = einops.parse_shape(cvol, '... c _ _')['c']
        dims = einops.parse_shape(cvol, 'c _ _')['c']
    imax = tf.argmax(cvol, axis=axis)

    # unravel_index
    imax = tf.cast(imax, tf.float32)
    q = tf.sqrt(tf.cast(dims, tf.float32))
    di = tf.floor(imax / q)
    dj = imax - di * q

    # delta from center
    di = di - (q - 1) / 2
    dj = dj - (q - 1) / 2

    return tf.stack([di, dj], axis=axis)


def flow_to_image(flow, data_format='channels_last'):
    if data_format == 'channels_last':
        axis = -1
    else:
        axis = -3

    # Map flow direction to hue.
    if data_format == 'channels_last':
        # ...hwc
        flo_ang = tf.atan2(flow[..., 1], flow[..., 0])
    else:
        # ...chw
        flo_ang = tf.atan2(flow[..., 1, :, :], flow[..., 0, :, :])
    # [-pi,pi) --> [0,1)
    h = (flo_ang + tf.constant(math.pi)) / tf.constant(2.0 * math.pi)

    # Map flow magnitude to saturation.
    # TODO(yycho0108): Potentially treat NaNs here?
    flo_mag = tf.norm(flow, axis=axis)  # ...HW

    smax = tf.reduce_max(flo_mag, axis=(-2, -1), keepdims=True)  # N11

    eps = tf.constant(1e-6)
    s = flo_mag / (smax + eps)

    # Value is always one.
    v = tf.ones_like(h)

    # NOTE(yycho0108): hsv axis here needs to be the last dimension.
    hsv = tf.stack([h, s, v], axis=-1)
    rgb = tf.image.hsv_to_rgb(hsv)
    if data_format == 'channels_first':
        # NOTE(yycho0108): Technically transpose() ... would be the correct operation,
        # but I figured it would be simpler for now to just stack-unstack.
        r, g, b = tf.unstack(rgb, axis=-1)
        img = tf.stack([r, g, b], axis=axis)
    else:
        img = rgb

    return img
