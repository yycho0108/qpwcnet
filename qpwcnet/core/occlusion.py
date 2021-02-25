#!/usr/bin/env python3

import tensorflow as tf
import einops

from qpwcnet.core.warp import tf_warp


def get_spatial_shape(x: tf.Tensor, data_format: str = None):
    # 3D pattern
    if data_format == 'channels_first':
        pattern = '_ h w'
        axis = -3
    else:
        pattern = 'h w _'
        axis = -1

    # Batch dimension
    if tf.rank(x) >= 4:
        is_batch = True
    if is_batch:
        pattern = 'n ' + pattern

    return einops.parse_shape(x, pattern)


def estimate_occlusion_map(flow: tf.Tensor,
                           data_format: str = None):
    """
    Estimate occlusion map from optical flow.
    Specifically, determine which pixels in the `next` frame cannot
    be determined based on the flow in the previous frame.
    flow specification:
    prv[i,j] = nxt[i+f[i,j,1], j+f[i,j,0]]

    TODO(ycho): non-stupid flow definition (major-minor)

    flow: (NHWC / NCHW tensor)
    """
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()

    if data_format == 'channels_first':
        axis = -3
    else:
        axis = -1

    shape = get_spatial_shape(flow, data_format)
    h, w = shape['h'], shape['w']
    i, j = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')

    #src = tf.concat([i, j], axis=axis)
    #dst = tf.cast(tf.round(src + flow), tf.int32)
    #oob = tf.reduce_any(tf.logical_or(dst <= 0, dst >= (h, w)), axis=axis)

    # di, dj = tf.unstack(flow, axis=axis)
    i = tf.cast(i, tf.float32)
    j = tf.cast(j, tf.float32)
    dj, di = tf.unstack(flow, axis=axis)
    i2, j2 = i + di, j + dj
    idx2 = tf.cast(tf.stack([i2, j2], axis=-1), tf.int32)

    # Clip idx2 just for our happiness
    idx2 = tf.clip_by_value(
        idx2,
        [0, 0],
        [h - 1, w - 1])

    # print('idx2.shape = {}'.format(idx2.shape))
    # print('dj.shape = {}'.format(dj.shape))
    # m = tf.SparseTensor(idx2, 1, dense_shape=tf.expand_dims(dj, axis).shape)
    # msk = tf.sparse.to_dense(m, default_value=0.0)
    # msk = tf.scatter_nd(idx2, tf.ones_like(dj), (1,) + dj.shape)
    oob = tf.reduce_any([i2 < 0, i2 >= h, j2 < 0, j2 >= w], axis=0)
    oob = tf.cast(oob, tf.float32)
    # Add batch dim to idx2
    b = einops.repeat(tf.range(shape['n']), 'n -> n h w c', h=h, w=w, c=1)
    idx2_wb = tf.concat([b, idx2], axis=-1)

    # NOTE(ycho): naive inverse flow.
    # works-ish. The assumption here : larger flow = closer flow
    # flow2 = -tf.tensor_scatter_nd_max(tf.zeros_like(flow), idx2_wb, flow)
    inv_flow = -tf_warp(flow, flow, data_format)
    dj, di = tf.unstack(inv_flow, axis=axis)
    i2, j2 = i + di, j + dj
    idx3 = tf.cast(tf.stack([i2, j2], axis=-1), tf.int32)
    idx3 = tf.clip_by_value(
        idx3,
        [0, 0],
        [h - 1, w - 1])
    idx3_wb = tf.concat([b, idx3], axis=-1)
    updates = tf.zeros_like(i2, dtype=tf.float32)
    map3 = tf.tensor_scatter_nd_min(tf.ones_like(oob), idx3_wb, updates)
    # map3 = 0 if value, 1 if no value

    oob = tf.maximum(oob, map3)

    ## print('flow2', flow2.shape)

    ## idx2 = tf.reshape(idx2, [-1, 2])

    ## outer_dims = len(idx2.shape) - 1  # == 3
    ## print('outer_dims', outer_dims)
    ## ix = idx2.shape[outer_dims]  # == 2
    ## print('ix', ix)

    ## len(updates.shape) - outer_dims == len(oob.shape) - ix
    ## len(updates.shape) - 3 == len(oob.shape) - 2
    ## len(updates.shape) - 3 == 4 - 2

    #updates = tf.zeros_like(i2, dtype=tf.float32)
    #print(oob.shape)  # 1,256,512
    #print(idx2.shape)  # 1,256,512,2
    #print(updates.shape)  # 1,256,512
    ## oob = True(1) if out-of-bounds
    ## idx2_wb = list of "valid"
    #oob = tf.tensor_scatter_nd_max(oob, idx2_wb, updates)
    return oob
