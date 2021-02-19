#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot

from qpwcnet.core.layers import _get_axis
from qpwcnet.core.layers import (
    Split, Upsample, Downsample, UpConv, Flow, UpFlow, lrelu, DownConv,
    FrameInterpolate)
from qpwcnet.core.mish import Mish, mish

from qpwcnet.core.agc import adaptive_clip_grad

from typing import Tuple

# from qpwcnet.core.f_layers import down_conv_block


def interpolator(feats_prv, feats_nxt, output_multiscale: bool = True):
    """ Frame interpolation stack. """
    data_format = tf.keras.backend.image_data_format()
    axis = _get_axis(data_format)  # feature axis

    # Create downsampled image pyramid.
    # Expect feats_prv/feats_nxt to be outputs from the encoder.
    # This means feats_prv[0] == img_prv, etc.
    imgs_prv = [feats_prv[0]]
    imgs_nxt = [feats_nxt[0]]
    for i in range(len(feats_prv) - 1):
        pool = Downsample()
        imgs_prv.append(pool(imgs_prv[-1]))
        imgs_nxt.append(pool(imgs_nxt[-1]))
    # Now, we have a downsampled image -
    # each of the same size as a corresponding feature layer.

    # Create flow layers pyramid.

    # Middle Frame is created from the composition of
    # the two images and the flow.
    # flo_01 = fwd, i.e. warp(nxt,flo_01)==prv
    # flo_10 = bwd, i.e. warp(prv,flo_10)==nxt
    # flo_10... path is only used(needed) for interpolator.
    n = len(feats_prv)
    flow = Flow()
    flo_01 = flow((feats_prv[-1], feats_nxt[-1]))
    flo_10 = flow((feats_nxt[-1], feats_prv[-1]))
    img = FrameInterpolate(up=False, name='img_{}'.format(n - 1))((
        imgs_prv[-1], imgs_nxt[-1], flo_01, flo_10))
    imgs = [img]

    # n-2 means skip the last one (for which we explicitly construct flow/img from scratch).
    # 0 means skip the final layer (and only apply upsampling).
    for i in range(n - 2, 0, -1):
        # Get inputs at current layer ...
        feat_prv = feats_prv[i]
        feat_nxt = feats_nxt[i]
        img_prv = imgs_prv[i]
        img_nxt = imgs_nxt[i]

        # Create layers at the current level.
        upsample = Upsample(scale=2.0)
        upflow = UpFlow()

        # Compute current stage motion block.
        # previous motion block + network features
        # NOTE(ycho): Unlike typical upsampling, also mulx2
        flo_01_u = upsample(flo_01)
        flo_10_u = upsample(flo_10)

        flo_01 = upflow((feat_prv, feat_nxt, flo_01_u))
        flo_10 = upflow((feat_nxt, feat_prv, flo_10_u))

        # Upsampled previous image + motion block +
        # downsampled input images
        img_u = Upsample(scale=1.0)(img)
        img = FrameInterpolate(up=True, name='img_{}'.format(i))((
            img_prv, img_nxt, flo_01, flo_10, img_u))
        imgs.append(img)

    # Final full-res img is ONLY upsampled.
    img = Upsample(scale=1.0, name='img_0')(img)
    imgs.append(img)

    if output_multiscale:
        return imgs
    # o.w. only return last img.
    return imgs[-1]


def encoder(inputs: tf.keras.Input, output_features: bool = False):
    """
    Feature computation encoder.
    Assumes inputs is a pair of images that has been
    concatenated in the `channels` axis.
    """
    data_format = tf.keras.backend.image_data_format()
    axis = _get_axis(data_format)

    # Split to two streams.
    img_prv, img_nxt = Split(2, axis=axis)(inputs)

    # Build out feature layers.
    # NOTE(ycho): Let's try to build this without batchnorm.
    # Maybe AGC will solve all our problems.
    layers = []
    for num_filters in [16, 32, 64, 128, 256]:
        layers.append(DownConv(num_filters, use_normalizer=False))

    # Apply feature layers...
    f = img_prv
    feats_prv = [f]
    for l in layers:
        f = l(f)
        feats_prv.append(f)

    f = img_nxt
    feats_nxt = [f]
    for l in layers:
        f = l(f)
        feats_nxt.append(f)

    if output_features:
        # all intermediate layers are included
        return (feats_prv, feats_nxt)
    else:
        return (feats_prv[-1], feats_nxt[-1])


def build_network(train=True, data_format='channels_first') -> tf.keras.Model:
    if data_format == 'channels_first':
        shape = (6, 256, 512)
        axis = 1
    elif data_format == 'channels_last':
        shape = (256, 512, 6)
        axis = 3
    else:
        raise ValueError('Unsupported data format : {}'.format(data_format))

    inputs = tf.keras.Input(shape=shape, dtype=tf.float32, name='inputs')
    img_prv, img_nxt = Split(2, axis=axis)(inputs)

    # Compute features.
    feat_layers = []
    for f in [16, 32, 64, 128, 256]:
        feat_layers.append(DownConv(f))

    feats_prv = []
    feats_nxt = []

    f = img_prv
    for l in feat_layers:
        f = l(f)
        feats_prv.append(f)

    f = img_nxt
    for l in feat_layers:
        f = l(f)
        feats_nxt.append(f)

    # Compute optical flow.
    flo = None
    flos = []
    for feat_prv, feat_nxt in zip(feats_prv[::-1], feats_nxt[::-1]):
        if flo is not None:
            # Compute upsampled flow from the previous layer.
            flo_u = Upsample(scale=2.0)(flo)

            # Compute the refined flow.
            args = (feat_prv, feat_nxt, flo_u)
            flo = UpFlow()(args)
        else:
            # Compute the first flow layer.
            args = (feat_prv, feat_nxt)
            flo = Flow()(args)
        flos.append(flo)

    # Compute final full-res optical flow.
    flo = Upsample(scale=2.0)(flo)
    flos.append(flo)

    if train:
        outputs = flos
    else:
        outputs = [flo]
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='qpwc_net')


def build_interpolator(input_shape: Tuple[int, int], *args, **kwargs):
    # input
    inputs = tf.keras.Input(
        shape=input_shape + (6,),
        dtype=tf.float32, name='inputs')
    # features
    feats_prv, feats_nxt = encoder(inputs, True)
    out = interpolator(feats_prv, feats_nxt, *args, **kwargs)
    return tf.keras.Model(inputs=inputs, outputs=out, name='qpwcnet-pretrain')


def main():
    net = build_network()
    net.summary()


if __name__ == '__main__':
    main()
