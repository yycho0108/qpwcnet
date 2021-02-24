#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa

from qpwcnet.core.warp import tf_warp
from qpwcnet.core.mish import Mish

from typing import Tuple


gamma = 0.00004


def lrelu(x):
    return tf.nn.leaky_relu(x, 0.1)


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
    """
    Compute CostVolume from composing tensorflow operations.
    Replacing CostVolumeV2() with CostVolume() lets us
    convert the model to tflite.
    """

    def __init__(
            self, search_range=4, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()

        self._config = {
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
        # assume prv.shape == nxt.shape
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
            self, search_range=4, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self._config = {
            'search_range': search_range,
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
    def __init__(self, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
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
        return tf_warp(img, flo, self.data_format)


class WarpV2(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
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
    def __init__(self, filters: int, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self._config = {
            'filters': filters,
        }

        axis = _get_axis(data_format)
        # self.norm = tf.keras.layers.BatchNormalization(axis=axis)
        self.conv_up = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=4, strides=2, padding='same',
            activation='Mish',
            kernel_regularizer=tf.keras.regularizers.l2(gamma),
            data_format=data_format)
        super().__init__(*args, **kwargs)

    def call(self, x, training=None):
        return self.conv_up(x)
        # return self.norm(self.conv_up(x), training=training)

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class OptFlow(tf.keras.layers.Layer):
    def __init__(self, filters: Tuple[int, ...] = (
            128, 64, 32, 16), *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.data_format = data_format
        axis = _get_axis(data_format)

        self._config = {
            'filters': filters
        }

        # Intermediate Features ...
        self.feat = []
        for f in filters:
            conv = tf.keras.layers.SeparableConv2D(
                filters=f, kernel_size=3, strides=1, padding='same',
                activation='Mish', use_bias=True, data_format=data_format)
            self.feat.append(conv)

        # Normalize right before computing the flow.
        self.norm = tf.keras.layers.BatchNormalization(axis=axis, fused=False)

        # Final flow with free scale
        self.flow = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=3,  # ?
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            data_format=data_format
        )
        # flow scale should be normalized wrt input shape
        self.scale = 1.0
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        shape = input_shape
        if self.data_format == 'channels_first':
            # NCHW -> 2,3
            h, w = shape[2], shape[3]
        elif self.data_format == 'channels_last':
            # NHWC -> 1,2
            h, w = shape[1], shape[2]
        else:
            raise ValueError(
                'Unsupported data format : {}'.format(data_format))
        self.scale = (float(h)**2 + float(w**2)) ** 0.5
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for conv in self.feat:
            x = conv(x)
        x = self.scale * self.flow(self.norm(x, training=training))
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FrameInterpolate(tf.keras.layers.Layer):
    def __init__(self, up: bool = False, *args, **kwargs):
        self._config = {
            'up': up
        }
        data_format = tf.keras.backend.image_data_format()
        self.up = up
        self.axis = _get_axis(data_format)
        self.warp = WarpV2()
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

        # Applying half-scale flow is valid-ish.
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

    def __init__(self, use_tfa: bool = True, *args, **kwargs):
        self._config = {
            'use_tfa': use_tfa
        }
        data_format = tf.keras.backend.image_data_format()
        self.flow = OptFlow()
        self.axis = _get_axis(data_format)

        if use_tfa:
            self.cost_volume = CostVolumeV2()
        else:
            self.cost_volume = CostVolume()

        super().__init__(*args, **kwargs)

    def call(self, inputs):
        prv, nxt = inputs
        cost = self.cost_volume((prv, nxt))
        feat = [cost, prv, nxt]
        feat = tf.concat(feat, axis=self.axis)
        # Consider additional procssing here.
        return self.flow(feat)

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UpFlow(tf.keras.layers.Layer):
    """
    Second optical flow estimation block.
    Refine/upsample the input optical flow.
    """

    def __init__(self, use_tfa: bool = True, *args, **kwargs):
        self._config = {
            'use_tfa': use_tfa
        }
        data_format = tf.keras.backend.image_data_format()
        self.flow = OptFlow()
        # self.warp = Warp()
        self.warp = WarpV2()
        if use_tfa:
            self.cost_volume = CostVolumeV2()
        else:
            self.cost_volume = CostVolume()
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

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DownConv(tf.keras.layers.Layer):
    """Conv+Mish+GroupNorm"""

    def __init__(self, num_filters,
                 use_normalizer: bool = True,
                 *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        axis = _get_axis(data_format)

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

        if self.use_normalizer:
            self.norm_a = tf.keras.layers.BatchNormalization(axis=axis)
            self.norm_aa = tf.keras.layers.BatchNormalization(axis=axis)
            self.norm_b = tf.keras.layers.BatchNormalization(axis=axis)
            #self.norm_a = tfa.layers.GroupNormalization(groups=4, axis=axis)
            #self.norm_aa = tfa.layers.GroupNormalization(groups=4, axis=axis)
            #self.norm_b = tfa.layers.GroupNormalization(groups=4, axis=axis)

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
