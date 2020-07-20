#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
from warp import tf_warp as warp


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
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)
    return cost_vol


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
    def __init__(self, filters, name='upsample', *args, **kwargs):
        self._config = {
            'filters': filters,
            'name': name
        }
        self.conv = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                    kernel_size=4,
                                                    strides=2,
                                                    padding='same',
                                                    activation=None,
                                                    name=name + '_conv'
                                                    )
        super().__init__(name=name, *args, **kwargs)

    def call(self, x):
        return tf.constant(2, dtype=tf.float32) * self.conv(x)

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Flow(tf.keras.layers.Layer):
    def __init__(self, name='flow', *args, **kwargs):
        self._config = {
            'name': name
        }
        self.flow = tf.keras.layers.Conv2D(filters=2,
                                           kernel_size=3,
                                           strides=1,
                                           padding='same'
                                           )

        super().__init__(name=name, *args, **kwargs)

    def call(self, inputs):
        # prv, nxt, flo):
        prv, nxt = inputs
        nxt_w = nxt
        cost = cost_volume_v2(prv, nxt)
        feat = [cost, prv, nxt]
        feat = tf.concat(feat, axis=-1)
        # consider additional procssing here ...
        return self.flow(feat)

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UpFlow(tf.keras.layers.Layer):
    def __init__(self, name='upflow', *args, **kwargs):
        self._config = {
            'name': name
        }
        self.flow = tf.keras.layers.Conv2D(filters=2,
                                           kernel_size=3,
                                           strides=1,
                                           padding='same'
                                           )
        super().__init__(name=name, *args, **kwargs)

    def call(self, inputs):
        # prv, nxt, flo):
        prv, nxt, flo = inputs
        feat = []
        if flo is None:
            nxt_w = nxt
            cost = cost_volume_v2(prv, nxt_w)
            feat = [cost, prv, nxt]
        else:
            nxt_w = warp(nxt, flo)
            feat.append(flo)
            cost = cost_volume_v2(prv, nxt_w)
            feat = [cost, prv, flo]
        feat = tf.concat(feat, axis=-1)
        # consider additional procssing here ...
        return self.flow(feat)

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
