#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot

from qpwcnet.core.layers import Split, Upsample, Flow, UpFlow, FeaturesLayer, lrelu, DownConv
# from mish import Mish, mish
from qpwcnet.core.mish import Mish, mish


def group_upconv(in1, groups, name='upconv'):
    # keras don't have an easy way of group conv so use old way
    with tf.compat.v1.variable_scope('pwcnet'):
        with tf.compat.v1.variable_scope(name):
            filterc = tf.compat.v1.get_variable(
                'filter_w', shape=[4, 4, 1, groups], dtype=tf.float32)
            shp = tf.shape(in1)
            output_shape = (shp[0], shp[1] * 2, shp[2] * 2, shp[3])
            return tf.nn.conv2d_transpose(in1, filterc, output_shape, strides=[1, 2, 2, 1])


def compute_features(img, train=True):
    feat = img

    convs = []
    out = []

    for f in [32, 64, 96, 128, 192]:
        # conv = tf.keras.layers.Conv2D(filters=f, kernel_size=3,
        #                              strides=2, activation=Mish(mish), padding='same')
        conv = tf.keras.layers.Conv2D(filters=f, kernel_size=3,
                                      strides=2, activation=None, padding='same',
                                      kernel_regularizer=tf.keras.regularizers.l2(
                                          l=0.0004)
                                      )
        act = tf.keras.layers.Activation('Mish')
        # norm = tf.keras.layers.BatchNormalization()
        norm = tfa.layers.GroupNormalization(groups=4, axis=3)
        feat = norm(act(conv(feat)), training=train)
        # feat = mish(feat)
        # feat = tf.keras.layers.LeakyReLU(0.1)(feat)
        convs.append(conv)
        out.append(feat)
    return convs, out


def build_network(train=True) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(256, 512, 6),
                            dtype=tf.float32, name='inputs')
    img_prv, img_nxt = Split(2, axis=-1)(inputs)

    feat_layers = []
    for f in [32, 64, 96, 128, 192]:
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

    # feat_module = FeaturesLayer()
    # convs_prv, feats_prv = compute_features(img_prv, train=train)
    # convs_nxt, feats_nxt = compute_features(img_nxt, train=train)
    # upsample = Upsample(2)

    # feats_prv = feat_module(img_prv)  # large->small
    # feats_nxt = feat_module(img_nxt)  # large->small

    flo = None
    count = 0
    flos = []
    for feat_prv, feat_nxt in zip(feats_prv[::-1], feats_nxt[::-1]):
        if flo is not None:
            # flo_u = Upsample(2)(flo)
            flo_h, flo_w = flo.shape[1:3]
            flo_u = tf.image.resize(flo, (2 * flo_h, 2*flo_w))
            args = [feat_prv, feat_nxt, flo_u]
            flo = UpFlow()(args)
        else:
            args = [feat_prv, feat_nxt]
            flo = Flow()(args)
        flos.append(flo)
        count += 1

    # Final full-res optical flow.
    flo = Upsample(2, name='upsample_{:02d}'.format(count))(flo)
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
