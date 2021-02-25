#!/usr/bin/env python3

"""
Copy of layers.py, but now all layers are plain objects instead.
"""

import tensorflow as tf
import tensorflow_addons as tfa
import einops

from qpwcnet.core.warp import tf_warp
from qpwcnet.core.mish import Mish

from typing import Tuple, List


gamma = 0.000004


def parse_image_shape(tensor: tf.Tensor,
                      data_format=None, blocklist: List[str] = []):
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    if data_format is 'channels_first':
        pattern = 'n c h w'
    else:
        pattern = 'n h w c'

    for block in blocklist:
        pattern.replace(block, '_')
    return einops.parse_shape(tensor, pattern)


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


class CostVolume:
    """
    Compute CostVolume from composing tensorflow operations.
    Replacing CostVolumeV2() with CostVolume() lets us
    convert the model to tflite.
    """

    def __init__(
            self, search_range=4, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.search_range = search_range
        self.data_format = data_format
        self.axis = _get_axis(data_format)
        # NOTE(yycho0108): channels_last
        self.pad = tf.keras.layers.ZeroPadding2D(
            self.search_range, data_format=data_format)
        self.h = None
        self.w = None

        super().__init__(*args, **kwargs)

    def __call__(self, inputs):
        prv, nxt = inputs

        shape = parse_image_shape(prv, self.data_format)
        h, w = shape['h'], shape['w']
        r = self.search_range

        d = r * 2 + 1

        pad_nxt = self.pad(nxt)

        cost_vol = []
        for i0 in range(0, d):
            for j0 in range(0, d):
                # roi = pad_nxt[:, i0:i0+h, j0:j0+w, -1]
                if self.data_format == 'channels_first':
                    roi = tf.slice(
                        pad_nxt, [0, 0, i0, j0], [-1, -1, h, w])
                elif self.data_format == 'channels_last':
                    roi = tf.slice(
                        pad_nxt, [0, i0, j0, 0], [-1, h, w, -1])
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


class CostVolumeV2:
    """
    Optimized routine with tfa.
    """

    def __init__(self, search_range=4, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.search_range = search_range
        self.corr = tfa.layers.optical_flow.CorrelationCost(
            1, search_range, 1, 1, search_range, data_format)
        super().__init__(*args, **kwargs)

    def __call__(self, inputs):
        prv, nxt = inputs
        cost_vol = self.corr([prv, nxt])
        out = tf.nn.leaky_relu(cost_vol, 0.1)
        return out


class Warp:
    def __init__(self, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.data_format = data_format
        super().__init__(*args, **kwargs)

    def __call__(self, inputs):
        img, flo = inputs
        return tf_warp(img, flo, self.data_format)


class WarpV2:
    """ Warp, but with tfa """

    def __init__(self, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.data_format = data_format
        self.warp = tf.keras.layers.Lambda(
            lambda a: tfa.image.dense_image_warp(a[0], a[1]))
        super().__init__(*args, **kwargs)

    def __call__(self, inputs):
        img, flo = inputs
        if self.data_format == 'channels_first':
            img = tf.transpose(img, (0, 2, 3, 1))
            flo = tf.transpose(flo, (0, 2, 3, 1))
            out = self.warp((img, -flo[..., ::-1]))
            # out = tfa.image.dense_image_warp(img, -flo[..., ::-1])
            out = tf.transpose(out, (0, 3, 1, 2))
        else:
            out = self.warp((img, -flo[..., ::-1]))
            # out = tfa.image.dense_image_warp(img, -flo[..., ::-1])
        return out


class Split:
    def __init__(self, num=2, axis=-1, *args, **kwargs):
        self.num = num
        self.axis = axis
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        return tf.split(x, self.num, self.axis)


class Downsample:
    def __init__(self, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.downsample = tf.keras.layers.AvgPool2D(
            data_format=data_format,
            pool_size=(2, 2), padding='same')
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        return self.downsample(x)


class Upsample:
    def __init__(self, scale: float = 1.0, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.scale = scale
        self.upsample = tf.keras.layers.UpSampling2D(
            data_format=data_format, interpolation='bilinear')
        super().__init__()
        # super().__init__(*args, **kwargs)

    def __call__(self, x):
        return tf.constant(self.scale, dtype=tf.float32) * self.upsample(x)


class UpConv:
    def __init__(self, filters: int, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        axis = _get_axis(data_format)
        # self.norm = tf.keras.layers.BatchNormalization(axis=axis)
        self.conv_up = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=4, strides=2, padding='same',
            activation='Mish',
            kernel_regularizer=tf.keras.regularizers.l2(gamma),
            data_format=data_format)
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        return self.conv_up(x)
        # return self.norm(self.conv_up(x), training=training)


class OptFlow:
    def __init__(self,
                 filters: Tuple[int, ...] = (128, 64, 32, 16),
                 scale: float = None,
                 *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.data_format = data_format
        axis = _get_axis(data_format)

        # Intermediate Features ...
        self.feat = []
        for i, f in enumerate(filters):
            conv = tf.keras.layers.SeparableConv2D(
                filters=f, kernel_size=3, strides=1, padding='same',
                activation='Mish', use_bias=True, data_format=data_format,
                # name='of_feat_{}'.format(i)
            )
            self.feat.append(conv)
        self.scale = scale

        # Normalize right before computing the flow.
        # NOTE(ycho): `fused` necessary for TFLITE conversion?
        # self.norm = tf.keras.layers.BatchNormalization(axis=axis, fused=False)

        # 1x1 conv for easier tflite conversion. yep, we're really doing this.
        self.conv = tf.keras.layers.Conv2D(
            filters=filters[-1],
            kernel_size=1, strides=1, padding='same', activation='Mish',
            use_bias=True, data_format=data_format)
        self.norm = tf.keras.layers.BatchNormalization(axis=axis, fused=False)

        # Final flow with free scale
        self.flow = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=3,  # ?
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            data_format=data_format,
            # name='of_flow'
        )
        super().__init__(*args, **kwargs)

    def __call__(self, inputs):
        shape = parse_image_shape(inputs)

        # NOTE(ycho): flow scale is normalized wrt input shape.
        if self.scale is None:
            scale = float(shape['h']**2 + shape['w']**2)**0.5
        else:
            # I guess this is useful when we're trying to load
            # an existing `scale`.
            scale = self.scale

        x = inputs
        for conv in self.feat:
            x = conv(x)
        f = self.flow(self.norm(self.conv(x)))
        x = scale * f
        return x


class FrameInterpolate:
    def __init__(self, up: bool = False, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.up = up
        self.axis = _get_axis(data_format)
        self.warp = WarpV2()
        # self.warp = Warp()
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
        super().__init__()
        # super().__init__(*args, **kwargs)

    def __call__(self, inputs):
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


class Flow:
    """
    First optical flow estimation block.
    """

    def __init__(self, use_tfa: bool = True, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.flow = OptFlow()
        self.axis = _get_axis(data_format)

        if use_tfa:
            self.cost_volume = CostVolumeV2()
        else:
            self.cost_volume = CostVolume()

        super().__init__(*args, **kwargs)

    def __call__(self, inputs):
        prv, nxt = inputs
        cost = self.cost_volume((prv, nxt))
        feat = [cost, prv, nxt]
        feat = tf.concat(feat, axis=self.axis)
        # Consider additional procssing here.
        return self.flow(feat)


class UpFlow:
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
        # NOTE(ycho): Consider also switching between Warp/WarpV2
        # based on `use_tfa`
        # self.warp = Warp()
        self.warp = WarpV2()

        if use_tfa:
            self.cost_volume = CostVolumeV2()
        else:
            self.cost_volume = CostVolume()
        self.axis = _get_axis(data_format)
        self.rename = tf.keras.layers.Lambda(lambda x: x)
        super().__init__(*args, **kwargs)

    def __call__(self, inputs):
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
        out = self.flow(feat)
        out = self.rename(out)
        return out


class DownConv:
    """Conv+Mish+GroupNorm"""

    def __init__(self, num_filters,
                 use_normalizer: bool = True,
                 *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        axis = _get_axis(data_format)
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

    def __call__(self, inputs, training=None):
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


class Flower:
    def __init__(self,
                 num_layers: int,
                 output_multiscale: bool = True,
                 use_tfa: bool = True,
                 ):
        self.num_layers = num_layers
        self.output_multiscale = output_multiscale
        self.use_tfa = use_tfa

        self.flow = Flow(use_tfa=use_tfa)
        self.upsamples = []
        self.upflows = []
        for i in range(num_layers):
            self.upsamples.append(Upsample(scale=2.0))
            self.upflows.append(UpFlow(use_tfa=use_tfa))
        self.upsamples.append(Upsample(sacle=2.0))

    def __call__(self, inputs, output_multiscale: bool = True,
                 use_tfa: bool = True):
        """ Frame interpolation stack. """
        (enc_prv, enc_nxt, decs_prv, decs_nxt) = inputs

        # flo_01 = fwd, i.e. warp(nxt,flo_01)==prv
        flo_01 = self.flow((enc_prv, enc_nxt))
        flos = [flo_01]

        for i in range(self.num_layers):
            # Get inputs at current layer ...
            dec_prv = decs_prv[i]
            dec_nxt = decs_nxt[i]

            # Compute current stage motion block.
            # previous motion block + network features
            # NOTE(ycho): Unlike typical upsampling, also mulx2
            flo_01_u = self.upsamples[i](flo_01)
            flo_01 = self.upflows[i]((dec_prv, dec_nxt, flo_01_u))
            flos.append(flo_01)

        # Final full-res flow is ONLY upsampled.
        flo_01 = self.upsamples[-1](flo_01)
        flos.append(flo_01)

        if self.output_multiscale:
            outputs = flos
        else:
            outputs = [flo_01]
        return outputs
