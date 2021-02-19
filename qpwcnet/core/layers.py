#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa

from qpwcnet.core.warp import get_pixel_value
from qpwcnet.core.mish import Mish


gamma = 0.0004


def lrelu(x):
    return tf.nn.leaky_relu(x, 0.1)


def cost_volume(prv, nxt, search_scale=1, data_format='channels_last'):
    corr = tfa.layers.optical_flow.CorrelationCost(
        1, 3 * search_scale, 1 * search_scale, 1 * search_scale, 3 *
        search_scale, data_format)(
        [prv, nxt])
    return lrelu(corr)


def _get_axis(data_format: str):
    """ Get channel axis from data format. """
    axis = None
    if data_format == 'channels_first':
        axis = 1
    elif data_format == 'channels_last':
        axis = 3
    else:
        raise ValueError(
            'Unsupported data format : {}'.format(data_format))
    return axis


class CostVolume(tf.keras.layers.Layer):
    def __init__(
            self, search_range=4, data_format='channels_first', *args, **
            kwargs):
        self._config = {
            'data_format': data_format,
            'search_range': search_range
        }
        self.search_range = search_range
        self.data_format = data_format
        self.axis = _get_axis(data_format)
        # NOTE(yycho0108): channels_last
        self.pad = tf.keras.layers.ZeroPadding2D(
            self.search_range, data_format=data_format)
        self.h = None
        self.w = None

        super().__init__(*args, **kwargs)

    def build(self, input_shapes):
        shape = input_shapes[0]
        if self.data_format == 'channels_first':
            self.h = shape[2]
            self.w = shape[3]
        elif self.data_format == 'channels_last':
            self.h = shape[1]
            self.w = shape[2]
        else:
            raise ValueError(
                'Unsupported data format : {}'.format(self.data_format))
        super().build(input_shapes)

    def call(self, inputs):
        prv, nxt = inputs
        r = self.search_range
        d = r * 2 + 1

        pad_nxt = self.pad(nxt)

        cost_vol = []
        for i0 in range(0, d):
            for j0 in range(0, d):
                # roi = pad_nxt[:, i0:i0+h, j0:j0+w, -1]
                if self.data_format == 'channels_first':
                    roi = tf.slice(
                        pad_nxt, [0, 0, i0, j0], [-1, -1, self.h, self.w])
                elif self.data_format == 'channels_last':
                    roi = tf.slice(
                        pad_nxt, [0, i0, j0, 0], [-1, self.h, self.w, -1])
                else:
                    raise ValueError(
                        'Unsupported data format : {}'.format(
                            self.data_format))

                cost = tf.reduce_mean(prv * roi, axis=self.axis, keepdims=True)
                cost_vol.append(cost)
        cost_vol = tf.concat(cost_vol, axis=self.axis)
        # I simply cannot think of a reason for this to exist.
        # Maybe this will prevent NaNs?
        out = tf.nn.leaky_relu(cost_vol, 0.1)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CostVolumeV2(tf.keras.layers.Layer):
    """
    Optimized routine with tfa.
    """

    def __init__(
            self, search_range=4, data_format='channels_first', *args, **
            kwargs):
        self._config = {
            'search_range': search_range,
            'data_format': data_format
        }
        self.search_range = search_range
        self.corr = tfa.layers.optical_flow.CorrelationCost(
            1, search_range, 1, 1, search_range, data_format)
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        prv, nxt = inputs
        cost_vol = self.corr([prv, nxt])
        out = tf.nn.leaky_relu(cost_vol, 0.1)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Warp(tf.keras.layers.Layer):

    def __init__(self, data_format='channels_first', *args, **kwargs):
        self._config = {
            'data_format': data_format
        }
        self.data_format = data_format
        self.axis = _get_axis(data_format)
        self.h = None
        self.w = None
        super().__init__(*args, **kwargs)

    def build(self, input_shapes):
        shape = input_shapes[0]
        if self.data_format == 'channels_first':
            self.h = shape[2]
            self.w = shape[3]
        elif self.data_format == 'channels_last':
            self.h = shape[1]
            self.w = shape[2]
        else:
            raise ValueError(
                'Unsupported data format : {}'.format(data_format))
        super().build(input_shapes)

    def call(self, inputs):
        img, flo = inputs

        # Create a 2D sampling grid.
        x, y = tf.meshgrid(tf.range(self.w), tf.range(self.h))

        # 2D grid + batch + depth dims.
        x = tf.expand_dims(x[None, ...], self.axis)  # 11HW
        y = tf.expand_dims(y[None, ...], self.axis)
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        grid_src = tf.concat([x, y], axis=self.axis)  # 12HW
        grid_dst = grid_src + flo  # B2HW
        x, y = tf.unstack(grid_dst, axis=self.axis)  # BHW

        max_y = tf.cast(self.h - 1, tf.int32)
        max_x = tf.cast(self.w - 1, tf.int32)

        x0 = x
        y0 = y
        x0 = tf.cast(x0, tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(y0, tf.int32)
        y1 = y0 + 1

        # Clip to range [0, H/W] to not violate img boundaries.
        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x)
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y)

        # get pixel value at corner coords
        if self.data_format == 'channels_first':
            # nchw -> nhwc
            img_nhwc = tf.transpose(img, (0, 2, 3, 1))
            Ia = get_pixel_value(img_nhwc, x0, y0)
            Ib = get_pixel_value(img_nhwc, x0, y1)
            Ic = get_pixel_value(img_nhwc, x1, y0)
            Id = get_pixel_value(img_nhwc, x1, y1)
            # nhwc -> nchw
            Ia, Ib, Ic, Id = [tf.transpose(x, (0, 3, 1, 2))
                              for x in (Ia, Ib, Ic, Id)]
        else:
            Ia = get_pixel_value(img, x0, y0)
            Ib = get_pixel_value(img, x0, y1)
            Ic = get_pixel_value(img, x1, y0)
            Id = get_pixel_value(img, x1, y1)

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
        wa = tf.expand_dims(wa, axis=self.axis)
        wb = tf.expand_dims(wb, axis=self.axis)
        wc = tf.expand_dims(wc, axis=self.axis)
        wd = tf.expand_dims(wd, axis=self.axis)

        # compute output
        out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class WarpV2(tf.keras.layers.Layer):
    def __init__(self, data_format='channels_first', *args, **kwargs):
        self._config = {
            'data_format': data_format
        }
        self.data_format = data_format
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        img, flo = inputs
        if self.data_format == 'channels_first':
            img = tf.transpose(img, (0, 2, 3, 1))
            flo = tf.transpose(flo, (0, 2, 3, 1))
            out = tfa.image.dense_image_warp(img, -flo[..., ::-1])
            out = tf.transpose(out, (0, 3, 1, 2))
        else:
            out = tfa.image.dense_image_warp(img, -flo[..., ::-1])
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ContextLayer(tf.keras.layers.Layer):
    def __init__(self, dilation_levels, *args, **kwargs):
        self._config = {
            'dilation_levels': dilation_levels
        }
        self.dilation_levels = dilation_levels
        super().__init__(*args, **kwargs)


class Split(tf.keras.layers.Layer):
    def __init__(self, num=2, axis=-1, *args, **kwargs):
        self._config = {
            'num': num,
            'axis': axis,
        }
        super().__init__(*args, **kwargs)

    def call(self, x):
        return tf.split(x, self._config['num'], self._config['axis'])

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Downsample(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.downsample = tf.keras.layers.AvgPool2D(
            data_format=data_format,
            pool_size=(2, 2), padding='same')
        super().__init__(*args, **kwargs)

    def call(self, x):
        return self.downsample(x)


class Upsample(tf.keras.layers.Layer):
    def __init__(self, scale: float = 1.0, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self._config = {
            'scale': scale
        }
        self.scale = scale
        self.upsample = tf.keras.layers.UpSampling2D(
            data_format=data_format, interpolation='bilinear')
        super().__init__(*args, **kwargs)

    def call(self, x):
        return tf.constant(self.scale, dtype=tf.float32) * self.upsample(x)

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UpConv(tf.keras.layers.Layer):
    def __init__(self, filters, data_format, *args, **kwargs):
        self._config = {
            'filters': filters,
            'data_format': data_format
        }
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=4, strides=2, padding='same',
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(gamma),
            data_format=data_format)
        super().__init__(*args, **kwargs)

    def call(self, x):
        return tf.constant(2, dtype=tf.float32) * self.conv(x)

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class OptFlow(tf.keras.layers.Layer):
    filters = [128, 64, 32, 16, 8, 2]

    def __init__(self, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.flow = []
        for i, f in enumerate(self.filters):
            is_flow = (i == (len(self.filters) - 1))
            activation = None if is_flow else 'Mish'
            use_bias = False if is_flow else True
            conv = tf.keras.layers.SeparableConv2D(
                filters=f, kernel_size=3, strides=1, padding='same', activation=activation,
                use_bias=use_bias,
                # depthwise_regularizer=tf.keras.regularizers.l2(gamma),
                # pointwise_regularizer=tf.keras.regularizers.l2(gamma),
                data_format=data_format
            )
            # conv = tf.keras.layers.Conv2D(
            # filters=f, kernel_size=3, strides=1, padding='same',
            # activation=activation)
            self.flow.append(conv)
        super().__init__(*args, **kwargs)

    # def build(self, input_shapes):
    #    shape = input_shapes[0]
    #    if self.data_format == 'channels_first':
    #        self.h = shape[2]
    #        self.w = shape[3]
    #    elif self.data_format == 'channels_last':
    #        self.h = shape[1]
    #        self.w = shape[2]
    #    else:
    #        raise ValueError(
    #            'Unsupported data format : {}'.format(self.data_format))
    #    super().build(input_shapes)

    def call(self, inputs):
        x = inputs
        for f in self.flow:
            x = f(x)
        return x


class FrameInterpolate(tf.keras.layers.Layer):
    def __init__(self, up: bool = False, *args, **kwargs):
        self._config = {
            'up': up
        }
        data_format = tf.keras.backend.image_data_format()
        self.up = up
        self.axis = _get_axis(data_format)
        self.warp = WarpV2(data_format=data_format)
        self.conv1 = tf.keras.layers.SeparableConv2D(
            filters=64, kernel_size=3, strides=1, padding='same',
            activation='Mish', use_bias=True, data_format=data_format)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            strides=1,
            activation=None,
            padding='same',
            data_format=data_format)
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        if self.up:
            prv, nxt, flo_01, flo_10, img_u = inputs
        else:
            prv, nxt, flo_01, flo_10, = inputs

        # Applying half-scale flow is valid-ish, assuming constant depth etc.
        nxt_w = self.warp((nxt, 0.5 * flo_01))
        prv_w = self.warp((prv, 0.5 * flo_10))

        if self.up:
            feats = tf.concat(
                [prv_w, nxt_w, flo_01, flo_10, img_u],
                axis=self.axis)
        else:
            feats = tf.concat([prv_w, nxt_w, flo_01, flo_10], axis=self.axis)
        return self.conv2(self.conv1(feats))

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Flow(tf.keras.layers.Layer):
    """
    First optical flow estimation block.
    """

    def __init__(self, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.flow = OptFlow()
        self.axis = _get_axis(data_format)
        # self.cost_volume = CostVolume(data_format=data_format)
        self.cost_volume = CostVolumeV2(data_format=data_format)

        super().__init__(*args, **kwargs)

    def call(self, inputs):
        prv, nxt = inputs
        cost = self.cost_volume((prv, nxt))
        feat = [cost, prv, nxt]
        feat = tf.concat(feat, axis=self.axis)
        # Consider additional procssing here.
        return self.flow(feat)


class UpFlow(tf.keras.layers.Layer):
    """
    Second optical flow estimation block.
    Refine/upsample the input optical flow.
    """

    def __init__(self, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.flow = OptFlow()
        # self.warp = Warp(data_format=data_format)
        self.warp = WarpV2(data_format=data_format)
        # self.cost_volume = CostVolume(data_format=data_format)
        self.cost_volume = CostVolumeV2(data_format=data_format)
        self.axis = _get_axis(data_format)
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        # prv, nxt, flo):
        prv, nxt, flo = inputs
        feat = []

        # nxt_w = warp(nxt, flo)

        # sintel = (x,y) == (j,i) == (minor,major)
        # image_warp = (y,x) == (i,j) == (major,minor)
        # nxt_w = tfa.image.dense_image_warp(nxt, -flo[...,::-1])
        # nxt_w = dense_image_warp(nxt, flo[..., ::-1])
        nxt_w = self.warp((nxt, flo))

        feat.append(flo)
        cost = self.cost_volume((prv, nxt_w))
        feat = [cost, prv, flo]
        feat = tf.concat(feat, axis=self.axis)
        # TODO(yycho0108):
        # Consider additional procssing here.
        return self.flow(feat)


class DownConv(tf.keras.layers.Layer):
    """Conv+Mish+GroupNorm"""

    def __init__(self, num_filters,
                 use_normalizer: bool = True,
                 *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self._config = {
            'num_filters': num_filters,
            'use_normalizer': use_normalizer
        }
        self.use_normalizer = use_normalizer
        # self.conv = tf.keras.layers.SeparableConv2D(filters=num_filters, kernel_size=3,
        # strides=2, activation='Mish', padding='same')

        self.conv_a = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=2,
            activation='Mish',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(gamma),
            data_format=data_format)

        self.conv_aa = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            activation='Mish',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(gamma),
            data_format=data_format)

        self.conv_b = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            activation='Mish',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(gamma),
            data_format=data_format)
        axis = _get_axis(data_format)
        if self.use_normalizer:
            self.norm_a = tfa.layers.GroupNormalization(groups=4, axis=axis)
            self.norm_aa = tfa.layers.GroupNormalization(groups=4, axis=axis)
            self.norm_b = tfa.layers.GroupNormalization(groups=4, axis=axis)

        super().__init__(*args, **kwargs)

    def call(self, inputs, training=None):
        x = inputs
        if self.use_normalizer:
            x = self.norm_a(self.conv_a(x), training=training)
            x = self.norm_aa(self.conv_aa(x), training=training)
            x = self.norm_b(self.conv_b(x), training=training)
        else:
            x = (self.conv_a(x))
            x = (self.conv_aa(x))
            x = (self.conv_b(x))
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
