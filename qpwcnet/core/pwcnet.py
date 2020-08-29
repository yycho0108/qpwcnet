#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot

from qpwcnet.core.layers import Split, Upsample, UpConv, Flow, UpFlow, lrelu, DownConv
# from mish import Mish, mish
from qpwcnet.core.mish import Mish, mish


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
    for f in [32, 64, 96, 128, 192]:
        feat_layers.append(DownConv(f, data_format))

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
            flo_u = Upsample()(flo)

            # Compute the refined flow.
            args = (feat_prv, feat_nxt, flo_u)
            flo = UpFlow(data_format)(args)
        else:
            # Compute the first flow layer.
            args = (feat_prv, feat_nxt)
            flo = Flow(data_format)(args)
        flos.append(flo)

    # Compute final full-res optical flow.
    flo = Upsample()(flo)
    flos.append(flo)

    if train:
        outputs = flos
    else:
        outputs = [flo]
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='qpwc_net')


def main():
    net = build_network()
    net.summary()


if __name__ == '__main__':
    main()
