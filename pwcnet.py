#!/usr/bin/env python3

import numpy as np
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import tensorflow_addons as tfa

from layers import Split, Upsample, Flow, UpFlow, FeaturesLayer, lrelu
from quantize import DelegateConvConfig
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry


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
        conv = tf.keras.layers.Conv2D(filters=f, kernel_size=3,
                                      strides=2, activation=tf.keras.activations.relu, padding='same')
        feat = conv(feat)
        # feat = tf.keras.layers.LeakyReLU(0.1)(feat)
        convs.append(conv)
        out.append(feat)

    return convs, out


def build_network():
    print('build')
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


q_cfg_map = {}
q_cfg_map[Flow] = DelegateConvConfig(['flow'])
q_cfg_map[UpFlow] = DelegateConvConfig(['flow'])
q_cfg_map[Upsample] = DelegateConvConfig(['conv'])
lrelu_cfg = default_8bit_quantize_registry.Default8BitActivationQuantizeConfig()
print('?', q_cfg_map)
# q_cfg_map[tf.keras.activations.swish]: default_8bit_quantize_registry.Default8BitActivationQuantizeConfig()


def quantize_annotate_layer(layer):
    L = tf.keras.layers
    # (FeaturesLayer, Flow, UpFlow, Upsample)
    if isinstance(layer, tuple(list(q_cfg_map.keys()))):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, q_cfg_map[layer.__class__])
    elif isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    elif isinstance(layer, tf.keras.layers.LeakyReLU):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, lrelu_cfg)
    else:
        print(isinstance(layer, tf.keras.layers.LeakyReLU))
        print(q_cfg_map.keys())
        print('not quantizing class = ', layer.__class__)
    return layer


def quantize_model(model):
    with tfmot.quantization.keras.quantize_scope({
        'FeaturesLayer': FeaturesLayer,
        'Flow': Flow,
        'UpFlow': UpFlow,
        'Upsample': Upsample,
        'Split': Split,
        'DelegateConvConfig': DelegateConvConfig,
        'swish': tf.keras.activations.swish
    }):
        if False:
            qfun = tfmot.quantization.keras.quantize_model
            qmodel = qfun(model)
        else:
            # annotated model
            amodel = tf.keras.models.clone_model(
                model,
                clone_function=quantize_annotate_layer
            )
            # q-aware model
            qmodel = tfmot.quantization.keras.quantize_apply(amodel)
            print('qmodel!')
        qmodel.compile(optimizer='adam',
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=[tf.keras.metrics.MeanSquaredError()])
    qmodel.summary()
    return qmodel


def to_tflite(model):
    cvt = tf.lite.TFLiteConverter.from_keras_model(model)
    cvt.optimizations = [tf.lite.Optimize.DEFAULT]
    return cvt.convert()


def main():
    net = build_network()
    net.summary()

    tf.keras.utils.plot_model(
        net,
        to_file="net.png",
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

    # single forward pass.
    dummy = np.zeros((1, 384, 512, 2), dtype=np.float32)
    out = net(dummy)

    qnet = quantize_model(net)
    net_tfl = to_tflite(qnet)
    # net_tfl = to_tflite(net)
    with open('./pwcnet.tflite', 'wb') as f:
        f.write(net_tfl)


if __name__ == '__main__':
    main()
