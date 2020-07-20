#!/usr/bin/env python3

import numpy as np
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import tensorflow_addons as tfa

from layers import Split, Upsample, Flow, UpFlow, FeaturesLayer, lrelu
from quantize import DelegateConvConfig
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry

# from mish import Mish, mish


def group_upconv(in1, groups, name='upconv'):
    # keras don't have an easy way of group conv so use old way
    with tf.compat.v1.variable_scope('pwcnet'):
        with tf.compat.v1.variable_scope(name):
            filterc = tf.compat.v1.get_variable(
                'filter_w', shape=[4, 4, 1, groups], dtype=tf.float32)
            shp = tf.shape(in1)
            output_shape = (shp[0], shp[1] * 2, shp[2] * 2, shp[3])
            return tf.nn.conv2d_transpose(in1, filterc, output_shape, strides=[1, 2, 2, 1])


def compute_features(img):
    feat = img

    convs = []
    out = []

    for f in [32, 64, 96, 128, 192]:
        # conv = tf.keras.layers.Conv2D(filters=f, kernel_size=3,
        #                              strides=2, activation=Mish(mish), padding='same')
        conv = tf.keras.layers.Conv2D(filters=f, kernel_size=3,
                                      strides=2, activation='relu', padding='same')
        feat = conv(feat)
        # feat = mish(feat)
        # feat = tf.keras.layers.LeakyReLU(0.1)(feat)
        convs.append(conv)
        out.append(feat)

    return convs, out


def build_network():
    inputs = tf.keras.Input(shape=(384, 512, 2), dtype=tf.float32)
    img_prv, img_nxt = Split(2, axis=-1)(inputs)

    # feat_module = FeaturesLayer()
    convs_prv, feats_prv = compute_features(img_prv)
    convs_nxt, feats_nxt = compute_features(img_nxt)

    upsample = Upsample(2)

    # feats_prv = feat_module(img_prv)  # large->small
    # feats_nxt = feat_module(img_nxt)  # large->small

    flo = None
    count = 0
    for feat_prv, feat_nxt in zip(feats_prv[::-1], feats_nxt[::-1]):
        if flo is not None:
            flo_u = upsample(flo)
        name = 'upflow_{:02d}'.format(count)
        if flo is not None:
            args = [feat_prv, feat_nxt, flo_u]
            flo = UpFlow(name=name)(args)
        else:
            args = [feat_prv, feat_nxt]
            flo = Flow(name=name)(args)
        count += 1
    outputs = [upsample(flo)]
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='qpwc_net')


def main():
    net = build_network()
    net.summary()


if __name__ == '__main__':
    main()
