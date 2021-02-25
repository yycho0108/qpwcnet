#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa

from qpwcnet.core.layers import _get_axis
from qpwcnet.core.non_layers import (
    Split,
    Upsample,
    Downsample,
    UpConv,
    Flow,
    UpFlow,
    DownConv,
    FrameInterpolate,
    Flower
)

#from qpwcnet.core.layers import (
#    Split, Upsample, Downsample, UpConv, Flow, UpFlow, DownConv,
#    FrameInterpolate)

from qpwcnet.core.mish import Mish, mish

from typing import Tuple


def flower(enc_prv, enc_nxt, decs_prv, decs_nxt,
           output_multiscale: bool = True,
           use_tfa: bool = True):
    """ Frame interpolation stack. """
    data_format = tf.keras.backend.image_data_format()
    axis = _get_axis(data_format)  # feature axis

    # How many encoding/decoding layers?
    n = len(decs_prv)

    # flo_01 = fwd, i.e. warp(nxt,flo_01)==prv
    flow = Flow(use_tfa=use_tfa)
    flo_01 = flow((enc_prv, enc_nxt))
    flos = [flo_01]

    for i in range(n):
        # Get inputs at current layer ...
        dec_prv = decs_prv[i]
        dec_nxt = decs_nxt[i]

        # Create layers at the current level.
        upsample = Upsample(scale=2.0)
        upflow = UpFlow(use_tfa=use_tfa)

        # Compute current stage motion block.
        # previous motion block + network features
        # NOTE(ycho): Unlike typical upsampling, also mulx2
        flo_01_u = upsample(flo_01)
        flo_01 = upflow((dec_prv, dec_nxt, flo_01_u))
        flos.append(flo_01)

    # Final full-res flow is ONLY upsampled.
    flo_01 = Upsample(scale=2.0)(flo_01)
    flos.append(flo_01)

    if output_multiscale:
        outputs = flos
    else:
        outputs = [flo_01]
    return outputs


def interpolator(img_prv, img_nxt,
                 decs_prv, decs_nxt,
                 flos_01, flos_10,
                 output_multiscale: bool = True,
                 use_tfa: bool = True):
    """ Frame interpolation stack. """
    data_format = tf.keras.backend.image_data_format()
    axis = _get_axis(data_format)  # feature axis

    # How many encoding/decoding layers?
    n = len(decs_prv)

    # Create downsampled image pyramid.
    # Expect feats_prv/feats_nxt to be outputs from the `encoder(...)`.
    # This means feats_prv[0] == img_prv, etc.
    imgs_prv = [img_prv]
    imgs_nxt = [img_nxt]
    for i in range(n + 1):
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
    img = FrameInterpolate(up=False, name='img_0')((
        imgs_prv[-1], imgs_nxt[-1], flos_01[0], flos_10[0]))
    imgs = [img]

    # n-2 means skip the last one (for which we explicitly construct flow/img from scratch).
    # 0 means skip the final layer (and only apply upsampling).
    for i in range(n):
        # Get inputs at current layer ...
        dec_prv = decs_prv[i]
        dec_nxt = decs_nxt[i]
        img_prv = imgs_prv[n - i]
        img_nxt = imgs_nxt[n - i]

        # Create layers at the current level.
        upsample = Upsample(scale=2.0)

        # Upsampled previous image + motion block +
        # downsampled input images
        img_u = Upsample(scale=1.0)(img)
        img = FrameInterpolate(up=True, name='img_{}'.format(i + 1))((
            dec_prv, dec_nxt, flos_01[i + 1], flos_10[i + 1], img_u))
        imgs.append(img)

    # Final full-res img is ONLY upsampled.
    img = Upsample(scale=1.0, name='img_{}'.format(n + 1))(img)
    imgs.append(img)

    if output_multiscale:
        return imgs
    # o.w. only return last img.
    return imgs[-1]


def encoder(img_prv, img_nxt, output_features: bool = False,
            train: bool = True):
    """
    Feature computation encoder.
    Assumes inputs is a pair of images that has been
    concatenated in the `channels` axis.
    """
    # Build out feature layers.
    # NOTE(ycho): Let's try to build this without batchnorm?
    # Maybe AGC will solve all our problems.
    layers = []
    for num_filters in [16, 32, 64, 128, 256]:
        conv = DownConv(num_filters, use_normalizer=False)
        if not train:
            conv.trainable = False
        layers.append(conv)

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


def decoder(encs_prv, encs_nxt, use_skip: bool = True,
            train: bool = True):
    data_format = tf.keras.backend.image_data_format()
    axis = _get_axis(data_format)
    # print('axis={}'.format(axis)) # 1?

    # build
    layers = []
    for num_filters in [128, 64, 32, 16]:
        # NOTE(ycho): does not include layer of equal size as input
        conv = UpConv(num_filters)
        if not train:
            conv.trainable = False
        layers.append(conv)

    # apply/prv
    f = encs_prv[-1]
    i = -2
    decs_prv = []
    for l in layers:
        f = l(f)
        if use_skip:
            f = tf.concat([f, encs_prv[i]], axis=axis)
            i -= 1
        decs_prv.append(f)

    # apply/nxt
    f = encs_nxt[-1]
    i = -2
    decs_nxt = []
    for l in layers:
        f = l(f)
        if use_skip:
            f = tf.concat([f, encs_nxt[i]], axis=axis)
            i -= 1
        decs_nxt.append(f)
    return (decs_prv, decs_nxt)


def build_network(train=True,
                  input_shape: Tuple[int, int] = (256, 512),
                  data_format=None,
                  use_tfa: bool = True,
                  ) -> tf.keras.Model:
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    # Input
    if data_format == 'channels_first':
        inputs = tf.keras.Input(
            shape=(6,) + input_shape,
            dtype=tf.float32, name='inputs')
    else:
        inputs = tf.keras.Input(
            shape=input_shape + (6,),
            dtype=tf.float32, name='inputs')

    # Split input.
    axis = _get_axis(data_format)
    img_prv, img_nxt = Split(2, axis=axis)(inputs)

    # hmm...
    encs_prv, encs_nxt = encoder(img_prv, img_nxt, True,
                                 train=True)
    decs_prv, decs_nxt = decoder(encs_prv, encs_nxt, True,
                                 train=True)

    outputs = flower(encs_prv[-1], encs_nxt[-1],
                     decs_prv, decs_nxt,
                     output_multiscale=train,
                     use_tfa=use_tfa
                     )

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='qpwc_net')
    return model


def build_interpolator(
        input_shape: Tuple[int, int],
        data_format=None,
        use_tfa: bool = True,
        *args, **kwargs):
    # input
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_first':
        inputs = tf.keras.Input(
            shape=(6,) + input_shape,
            dtype=tf.float32, name='inputs')
    else:
        inputs = tf.keras.Input(
            shape=input_shape + (6,),
            dtype=tf.float32, name='inputs')

    # Split input.
    axis = _get_axis(data_format)
    img_prv, img_nxt = Split(2, axis=axis)(inputs)

    encs_prv, encs_nxt = encoder(img_prv, img_nxt, True)
    decs_prv, decs_nxt = decoder(encs_prv, encs_nxt, True)
    flower_block = Flower(
        len(decs_prv),
        output_multiscale=True,
        use_tfa=use_tfa)
    flows_01 = flower_block((encs_nxt[-1], encs_prv[-1], decs_nxt, decs_prv))

    # ^^^^ ALL code blocks above must EXACTLY match
    # build_network() in order for the transfer to work.
    # this is because we are not very meticulous about
    # bookkeeping layer correspondences.
    flows_10 = flower_block((encs_prv[-1], encs_nxt[-1], decs_prv, decs_nxt))

    outputs = interpolator(
        img_prv, img_nxt,
        decs_prv, decs_nxt,
        flows_01, flows_10,
        use_tfa=use_tfa, *args, **kwargs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='qpwc_net')


def main():
    net = build_network()
    net.summary()


if __name__ == '__main__':
    main()
