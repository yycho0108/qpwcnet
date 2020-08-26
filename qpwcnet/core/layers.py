#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa

from qpwcnet.core.warp import dense_image_warp, get_pixel_value
from qpwcnet.core.mish import Mish


def lrelu(x):
    return tf.nn.leaky_relu(x, 0.1)


def cost_volume(prv, nxt, search_scale=1, data_format='channels_last'):
    corr = tfa.layers.optical_flow.CorrelationCost(
        1, 3*search_scale, 1*search_scale, 1*search_scale, 3*search_scale, data_format)([prv, nxt])
    return lrelu(corr)


def cost_volume_v2(prv, nxt, search_range=4, name='cost_volume'):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        prv: Level of the feature pyramid of Image1
        nxt: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(nxt, [[0, 0], [search_range, search_range], [
                        search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(prv))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(prv * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    # I simply cannot think of a reason for this to exist.
    # cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)
    return cost_vol


class CostVolume(tf.keras.layers.Layer):
    def __init__(self, search_range=4, *args, **kwargs):
        self._config = {
            'search_range': search_range
        }
        self.search_range = search_range
        # NOTE(yycho0108): channels_last
        self.pad = tf.keras.layers.ZeroPadding2D(self.search_range)

        super().__init__(*args, **kwargs)

    # def build(self, input_shapes):

    def call(self, prv, nxt):
        r = self.search_range
        d = r * 2 + 1
        pad_nxt = self.pad(nxt)
        _, h, w, _ = tf.unstack(tf.shape(prv))

        cost_vol = []
        for i0 in range(0, d):
            for j0 in range(0, d):
                #roi = pad_nxt[:, i0:i0+h, j0:j0+w, -1]
                roi = tf.slice(pad_nxt, [0, i0, j0, 0], [-1, h, w, -1])
                cost = tf.reduce_mean(prv * roi, axis=3, keepdims=True)
                cost_vol.append(cost)
        cost_vol = tf.concat(cost_vol, axis=3)
        # I simply cannot think of a reason for this to exist.
        # cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)
        return cost_vol

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Warp(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        img, flo = inputs

        # Create a 2D sampling grid.
        W, H = tf.cast(tf.shape(img)[2], tf.float32), tf.cast(
            tf.shape(img)[1], tf.float32)
        x, y = tf.meshgrid(tf.range(W), tf.range(H))

        # 2D grid + batch + depth dims
        x = x[None, ..., None]
        y = y[None, ..., None]
        x = tf.cast(x, tf.float32)  # is this necessary?
        y = tf.cast(y, tf.float32)

        grid_src = tf.concat([x, y], axis=3)
        grid_dst = grid_src + flo
        x = grid_dst[:, :, :, 0]
        y = grid_dst[:, :, :, 1]

        max_y = tf.cast(H - 1, tf.int32)
        max_x = tf.cast(W - 1, tf.int32)
        zero = tf.zeros([], dtype=tf.int32)

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
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output
        out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

        return out


class ContextLayer(tf.keras.layers.Layer):
    def __init__(self, dilation_levels, *args, **kwargs):
        self._config = {
            'dilation_levels': dilation_levels
        }
        self.dilation_levels = dilation_levels
        super().__init__(*args, **kwargs)


class Split(tf.keras.layers.Layer):
    def __init__(self, num=2, axis=-1, name='split', *args, **kwargs):
        self._config = {
            'num': num,
            'axis': axis,
            'name': name
        }
        super().__init__(name=name, *args, **kwargs)

    def call(self, x):
        return tf.split(x, self._config['num'], self._config['axis'])

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Upsample(tf.keras.layers.Layer):
    def __init__(self, filters, *args, **kwargs):
        self._config = {
            'filters': filters,
        }
        self.conv = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                    kernel_size=4,
                                                    strides=2,
                                                    padding='same',
                                                    activation=None,
                                                    )
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
    filters = [128, 128, 96, 64, 32, 2]

    def __init__(self, *args, **kwargs):
        self.flow = []
        for i, f in enumerate(self.filters):
            is_flow = (i == (len(self.filters) - 1))
            activation = None if is_flow else 'Mish'
            use_bias = False if is_flow else True
            conv = tf.keras.layers.SeparableConv2D(
                filters=f, kernel_size=3, strides=1, padding='same', activation=activation,
                use_bias=use_bias)
            # conv = tf.keras.layers.Conv2D(
            #    filters=f, kernel_size=3, strides=1, padding='same', activation=activation)
            self.flow.append(conv)
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        x = inputs
        for f in self.flow:
            x = f(x)
        return x


class Flow(tf.keras.layers.Layer):
    """
    First optical flow estimation block.
    """

    def __init__(self, *args, **kwargs):
        self.flow = OptFlow()
        self.cost_volume = CostVolume()

        super().__init__(*args, **kwargs)

    def call(self, inputs):
        # prv, nxt, flo):
        prv, nxt = inputs
        cost = self.cost_volume(prv, nxt)
        feat = [cost, prv, nxt]
        feat = tf.concat(feat, axis=-1)
        # consider additional procssing here ...
        return self.flow(feat)


class UpFlow(tf.keras.layers.Layer):
    """
    Second optical flow estimation block.
    Refine/upsample the input optical flow.
    """

    def __init__(self, *args, **kwargs):
        self.flow = OptFlow()
        self.warp = Warp()
        self.cost_volume = CostVolume()
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        # prv, nxt, flo):
        prv, nxt, flo = inputs
        feat = []

        # nxt_w = warp(nxt, flo)

        # NOTE(yycho0108): [::-1] is required due to mismatch in convention.
        # sintel = (x,y) == (j,i) == (minor,major)
        # image_warp = (y,x) == (i,j) == (major,minor)
        # nxt_w = tfa.image.dense_image_warp(nxt, -flo[...,::-1])
        # nxt_w = dense_image_warp(nxt, flo[..., ::-1])
        nxt_w = self.warp((nxt, flo))

        feat.append(flo)
        cost = self.cost_volume(prv, nxt_w)
        feat = [cost, prv, flo]
        feat = tf.concat(feat, axis=-1)
        # consider additional procssing here ...
        return self.flow(feat)


class DownConv(tf.keras.layers.Layer):
    """Conv+Mish+GroupNorm"""

    def __init__(self, num_filters, *args, **kwargs):
        self._config = {
            'num_filters': num_filters,
        }
        # self.conv = tf.keras.layers.SeparableConv2D(filters=num_filters, kernel_size=3,
        #                                            strides=2, activation='Mish', padding='same')
        self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3,
                                           strides=2, activation='Mish', padding='same')
        self.norm = tfa.layers.GroupNormalization(groups=4, axis=3)
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=None):
        return self.norm(self.conv(inputs), training=training)

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FeaturesLayer(tf.keras.layers.Layer):
    def __init__(self, name='features', *args, **kwargs):
        self._config = {
            'name': name
        }
        self.convs = []
        for f in [32, 64, 96, 128, 192]:
            conv = tf.keras.layers.Conv2D(filters=f, kernel_size=3,
                                          strides=2, activation=tf.keras.activations.swish, padding='same')
            self.convs.append(conv)
        super().__init__(*args, **kwargs)

    def call(self, x):
        out = []
        for c in self.convs:
            x = c(x)
            out.append(x)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
