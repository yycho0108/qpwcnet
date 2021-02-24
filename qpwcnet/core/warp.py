#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
import einops


def get_pixel_value(
        img, x, y, data_format=None):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()

    if data_format == 'channels_first':
        axis = -3
    else:
        axis = -1

    batch_dims = tf.rank(img) - 3

    # NOTE(ycho): HAS TO BE -1 due to tensorflow constraints.
    # indices[-1]
    indices = tf.stack([y, x], axis=-1)

    if data_format == 'channels_first':
        img = einops.rearrange(img, '... c h w -> ... h w c')
    out = tf.gather_nd(img, indices, batch_dims=batch_dims)
    if data_format == 'channels_first':
        out = einops.rearrange(out, '... h w c -> ... c h w')

    #print(x.shape) # 1,256,512
    #print(y.shape) # 1,256,512
    #print(indices.shape) # 1,256,512,2
    #print(img.shape) #1,3,256,512
    #out = tf.gather_nd(img, indices, batch_dims=batch_dims)
    return out
    #shape = tf.shape(x)
    #batch_size = shape[0]
    #height = shape[1]
    #width = shape[2]

    # batch_idx = tf.range(0, batch_size)[:, None, None, None]  # B111
    # if axis == 1:
    #    b = tf.tile(batch_idx, (1, 1, height, width))  # B1HW
    # else:
    #    b = tf.tile(batch_idx, (1, height, width, 1))  # BHW1

    #indices = tf.stack([b, ALL, y, x], axis)
    # return tf.gather_nd(img, indices)


def tf_warp(img, flow, data_format=None):
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    # 3D pattern
    if data_format == 'channels_first':
        pattern = '_ h w'
        axis = -3
    else:
        pattern = 'h w _'
        axis = -1

    # Check if batched ...
    if tf.rank(flow) >= 4:
        is_batch = True

    if is_batch:
        pattern = '_ ' + pattern

    shape = einops.parse_shape(img, pattern)
    W, H = shape['w'], shape['h']

    # Compute grid coordinates.
    x, y = tf.meshgrid(tf.range(W), tf.range(H))

    # Add channel dims
    x = tf.expand_dims(x, axis=axis)
    y = tf.expand_dims(y, axis=axis)

    # Add batch dims
    if is_batch:
        x = tf.expand_dims(x, axis=0)
        y = tf.expand_dims(y, axis=0)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    grid = tf.concat([x, y], axis=axis)

    flows = grid + flow
    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.constant(0, dtype=tf.int32)

    # Deal with individual components
    if data_format == 'channels_first':
        x, y = einops.rearrange(flows, '... c h w -> c ... h w', c=2)
    else:
        x, y = einops.rearrange(flows, '... h w c -> c ... h w', c=2)

    x0 = x
    y0 = y
    x0 = tf.cast(x0, tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(y0, tf.int32)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0, data_format)
    Ib = get_pixel_value(img, x0, y1, data_format)
    Ic = get_pixel_value(img, x1, y0, data_format)
    Id = get_pixel_value(img, x1, y1, data_format)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=axis)
    wb = tf.expand_dims(wb, axis=axis)
    wc = tf.expand_dims(wc, axis=axis)
    wd = tf.expand_dims(wd, axis=axis)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


def dense_image_warp(image, flow, name=None) -> tf.Tensor:
    """Image warping using per-pixel flow vectors.
    Apply a non-linear warp to the image, where the warp is specified by a
    dense flow field of offset vectors that define the correspondences of
    pixel values in the output image back to locations in the source image.
    Specifically, the pixel value at output[b, j, i, c] is
    images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].
    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
    of the image, we use the nearest pixel values at the image boundary.
    PLEASE NOTE: The definition of the flow field above is different from that
    of optical flow. This function expects the negative forward flow from
    output image to source image. Given two images `I_1` and `I_2` and the
    optical flow `F_12` from `I_1` to `I_2`, the image `I_1` can be
    reconstructed by `I_1_rec = dense_image_warp(I_2, -F_12)`.
    Args:
      image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
      flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
      name: A name for the operation (optional).
      Note that image and flow can be of type tf.half, tf.float32, or
      tf.float64, and do not necessarily have to be the same type.
    Returns:
      A 4-D float `Tensor` with shape`[batch, height, width, channels]`
        and same type as input image.
    Raises:
      ValueError: if height < 2 or width < 2 or the inputs have the wrong
        number of dimensions.
    """
    with tf.name_scope(name or "dense_image_warp"):
        image = tf.convert_to_tensor(image)
        flow = tf.convert_to_tensor(flow)
        batch_size, height, width, channels = (
            tf.shape(image)[0],
            tf.shape(image)[1],
            tf.shape(image)[2],
            tf.shape(image)[3],
        )

        # The flow is defined on the image grid. Turn the flow into a list of query
        # points in the grid space.
        grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
        stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
        batched_grid = tf.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid + flow
        query_points_flattened = tf.reshape(
            query_points_on_grid, [batch_size, height * width, 2]
        )
        # Compute values at the query points, then reshape the result back to the
        # image grid.
        interpolated = tfa.image.interpolate_bilinear(
            image, query_points_flattened)
        interpolated = tf.reshape(
            interpolated, [batch_size, height, width, channels])
        return interpolated
