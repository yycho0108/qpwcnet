#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import inspect
from typing import Callable
from functools import partial

from qpwcnet.core.pwcnet import build_flower, build_interpolator
from qpwcnet.core.layers import *
from qpwcnet.core.mish import Mish, mish  # hmm...
from qpwcnet.train.util import load_weights
# from qpwcnet.core.quantize import DelegateConvConfig
# , CustomQuantizeScheme, replace_layer
from qpwcnet.core.quantize.quantize import RecursiveDelegateConfig, replace_layer
from qpwcnet.core.quantize.transform import Custom8BitQuantizeScheme
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation

# Alias for sanity preservation
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope

if False:
    q_cfg_map = {}
    q_cfg_map[Flow] = DelegateConvConfig(['flow'])
    q_cfg_map[UpFlow] = DelegateConvConfig(['flow'])
    # q_cfg_map[Upsample] = DelegateConvConfig(['conv'])
    lrelu_cfg = default_8bit_quantize_registry.Default8BitActivationQuantizeConfig()
    #print('?', q_cfg_map)
    #q_cfg_map[tf.keras.activations.swish]: default_8bit_quantize_registry.Default8BitActivationQuantizeConfig()
    #q_cfg_map[Mish]: default_8bit_quantize_registry.Default8BitActivationQuantizeConfig()
    # q_cfg_map[Mish(
    # mish)]:
    # default_8bit_quantize_registry.Default8BitActivationQuantizeConfig()


# quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer


#def deep_quantize_annotate_layer(
#        layer,
#        anno_fn: Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer],
#        is_leaf: Callable[[tf.keras.layers.Layer], bool]):
#    """
#    quantize_annotate_layer() analogue that works on nested Layers.
#    """
#    out = anno_fn(layer)
#    sub_layers = layer._flatten_layers(recursive=False, include_self=False)
#    print('l{}/ sl{}'.format(layer, sub_layers))
#    if not is_leaf(layer):
#        for sub_layer in sub_layers:
#            new_layer = deep_quantize_annotate_layer(sub_layer, anno_fn)
#            # Would setattr work? Or do we need to pull other tricks?
#            setattr(out, name, new_layer)
#    return out


def quantize_annotate_layer(layer):
    # do-nothing layers
    if isinstance(
            layer, (tf.python.keras.engine.base_layer.TensorFlowOpLayer)):
        # NOTE(ycho): Currently only `concat_* in `TensorFlowOpLayer`
        # so it's safe to do this. o.w. we might want to do something smarter.
        return layer

    # no-idea-what-to-do layers
    if isinstance(
            layer,
            (tfa.layers.optical_flow.CorrelationCost, tf.keras.layers.Lambda)):
        # Actually, I think this should just be replaced by an
        # equivalent (non-custom) op.
        return layer

    # Will be handled separately.
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        # Actually, I think this should just be replaced by an
        # equivalent (non-custom) op.
        return layer

    return tfmot.quantization.keras.quantize_annotate_layer(layer)

    # recursive composition layers
    if isinstance(layer, (UpConv,
                          DownConv,
                          #Flow, # hmm...
                          #UpFlow,
                          #OptFlow,
                          FrameInterpolate
                          )):
        return tfmot.quantization.keras.quantize_annotate_layer(
            layer, RecursiveDelegateConfig())

    # specialize on Mish
    if isinstance(layer, Mish):
        return tfmot.quantization.keras.quantize_annotate_layer(
            layer, default_8bit_quantize_registry.Default8BitActivationQuantizeConfig())

    # fallback
    print('fallback = {}({})'.format(layer, layer.name))
    return tfmot.quantization.keras.quantize_annotate_layer(layer)

    #L = tf.keras.layers
    #if isinstance(layer, tuple(list(q_cfg_map.keys()))):
    #    return tfmot.quantization.keras.quantize_annotate_layer(
    #        layer, q_cfg_map[layer.__class__])
    #elif isinstance(layer, tf.keras.layers.Conv2D):
    #    return tfmot.quantization.keras.quantize_annotate_layer(layer)
    #elif isinstance(layer, tf.keras.layers.LeakyReLU):
    #    return tfmot.quantization.keras.quantize_annotate_layer(
    #        layer, lrelu_cfg)
    #else:
    #    # print(isinstance(layer, tf.keras.layers.LeakyReLU))
    #    # print(q_cfg_map.keys())
    #    print('not quantizing class = ', layer.__class__)
    #return layer


# Yep, we're really doing this.
quantize_aware_activation.QuantizeAwareActivation._POST_QUANT_ACTIVATIONS = frozenset({
    *quantize_aware_activation.QuantizeAwareActivation._POST_QUANT_ACTIVATIONS, 'Mish', 'mish'})

#class QuantizeAwareMish(quantize_aware_activation.QuantizeAwareActivation):
#  _POST_QUANT_ACTIVATIONS = frozenset({'linear', 'relu', 'Mish'})
#  def __init__(self, *args, **kwargs):
#      super().__init__(*args, **kwargs)


def replace_predicate(layer: tf.keras.layers.Layer):
    return True


def replace_factory(layer: tf.keras.layers.Layer, name: str = None):
    cfg = layer.get_config()
    prv_name = cfg.pop('name')  # remove name ...
    if name is None:
        name = '{}_replaced'.format(layer.name)
    cfg['name'] = name
    new_layer = layer.from_config(cfg)
    print('{} -> {}'.format(layer, new_layer))
    return new_layer


def quantize_model(model: tf.keras.Model):

    #_anno = partial(deep_quantize_annotate_layer,
    #                anno_fn=quantize_annotate_layer,
    #                is_leaf=lambda _: False)
    _anno = quantize_annotate_layer

    with tfmot.quantization.keras.quantize_scope({
        # 'Split': Split,
        # 'Upsample': Upsample,
        # 'Downsample': Downsample,
        # 'UpConv': UpConv,
        # 'Flow': Flow,
        # 'UpFlow': UpFlow,
        # 'DownConv': DownConv,
        # 'OptFlow': OptFlow,
        # 'FrameInterpolate': FrameInterpolate,
        # 'RecursiveDelegateConfig': RecursiveDelegateConfig,
        'Mish': Mish(mish),
        'mish': Mish(mish),
        # 'CostVolumeV2': CostVolumeV2,
        # 'lrelu' : lrelu,
        # 'DelegateConvConfig': DelegateConvConfig,
        # 'swish': tf.keras.activations.swish,
        # 'mish': Mish(mish),
    }):
        # Hmm .....
        #cloned_model = tf.keras.models.clone_model(model,
        #        clone_function = lambda l
        #        )

        #new_input = tf.keras.Input((6, 256, 512))
        #cloned_model = tf.keras.Model(
        #    inputs=[new_input],
        #    outputs=model.call(new_input))

        #replaced_model = cloned_model

        #replaced_model = replace_layer(cloned_model,
        #                               replace_predicate,
        #                               replace_factory)
        # OPT 1.a ) default from package
        # annotated_model = quantize_annotate_model(model)
        # OPT 1.b ) use my own fun

        print('<annotate>')
        annotated_model = tf.keras.models.clone_model(
            model,
            clone_function=_anno
        )
        print('</annotate>')

        print('<quantize>')
        quantized_model = tfmot.quantization.keras.quantize_apply(
            annotated_model,
            # Custom8BitQuantizeScheme()
        )
        print('</quantize>')

        print('<compile>')
        quantized_model.compile(optimizer='adam')  # hmm
        print('</compile>')
    quantized_model.summary()
    return quantized_model


def to_tflite(model):
    cvt = tf.lite.TFLiteConverter.from_keras_model(model)
    cvt.allow_custom_ops = False  # Needed for tfa ops, apparently.
    cvt.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    cvt.optimizations = [tf.lite.Optimize.DEFAULT]
    return cvt.convert()


def main():
    # NOTE(ycho): NEEDED to avoid segfaults. Why? I don't know.
    # In other words, `channels_first` does NOT work.
    # Before attempting tflite conversion, modify data format to
    # `channels_last`. Loading weights, etc., should still work.
    tf.keras.backend.set_image_data_format('channels_last')

    if True:
        model = build_flower(
            train=False, input_shape=(
                256, 512), use_tfa=False)
        load_weights(model, '/tmp/pwc/run/017/model.h5')
        print('output = {}'.format(model.output.shape))
    else:
        model = build_interpolator(
            input_shape=(256, 512),
            data_format=tf.keras.backend.image_data_format(),
            use_tfa=False
        )
        load_weights(model, '/tmp/pwc/run/007/model.h5')

        # NOTE(ycho): Extract the flow01 component of interpolator model.
        model = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer('lambda_11').get_output_at(0)
        )

    # model.summary()

    # Single forward pass.
    # dummy = np.zeros((1, 6, 256, 512), dtype=np.float32)
    dummy = np.zeros((1, 256, 512, 6), dtype=np.float32)
    out = model(dummy)

    quantized_model = quantize_model(model)
    tflite_model = to_tflite(quantized_model)
    with open('/tmp/qpwcnet.tflite', 'wb') as f:
       f.write(tflite_model)


if __name__ == '__main__':
    main()
