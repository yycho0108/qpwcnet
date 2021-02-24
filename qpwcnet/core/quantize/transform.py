#!/usr/bin/env python3

from qpwcnet.core.layers import Flow, OptFlow
from collections import OrderedDict
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_transforms
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_transforms import _has_custom_quantize_config
from tensorflow_model_optimization.python.core.quantization.keras import quantize_scheme
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry

LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern


class OptFlowQuantize(transforms.Transform):
    """Break SeparableConv into a DepthwiseConv and Conv layer.

    SeparableConv is a composition of a DepthwiseConv and a Conv layer. For the
    purpose of quantization, a FQ operation needs to be placed between the output
    of DepthwiseConv and the following Conv.

    This is needed since there is a dynamic tensor in between the two layers, and
    it's range information needs to be captured by the FakeQuant op to ensure
    full int8 quantization of the layers is possible.

    Splitting the layer into 2 ensures that each individual layer is handled
    correctly with respect to quantization.
    """

    def pattern(self):
        return LayerPattern('OptFlow',
                            inputs=[LayerPattern('.*')])

    def replacement(self, match_layer):
        if _has_custom_quantize_config(match_layer):
            return match_layer

        print([(w, v.shape) for (w, v) in match_layer.weights.items()])
        ref_layer = OptFlow.from_config(match_layer.layer['config'])

        """
        x = inputs
        for conv in self.feat:
            x = conv(x)
        x = self.scale * self.flow(self.norm(x, training=training))
        """

        wlist = list(match_layer.weights.items())
        offset = 0
        for i, conv in enumerate(ref_layer.feat):
            cfg = tf.keras.layers.serialize(conv)

            # NOTE(ycho): maybe this doesn't even matter... idk
            cfg['name'] = conv.name

            # Build weights based on name.
            weights = OrderedDict()
            for k, w in wlist:
                if conv.name in k:
                    weights[k] = w

            node = LayerNode(
                cfg, weights=weights, input_layers=(
                    match_layer.input_layers if i <= 0 else [node]), metadata={
                    'quantize_config': None})
            offset += 3

        # Norm part will be auto taken care of by pattern matching later
        # (hopefully) since it comes right after conv.
        norm = ref_layer.norm
        cfg = tf.keras.layers.serialize(norm)
        cfg['name'] = norm.name
        weights = OrderedDict()
        for k, w in wlist:
            if norm.name in k:
                weights[k] = w
        node = LayerNode(cfg, weights=weights,
                         input_layers=[node],
                         metadata={'quantize_config': None})

        # Apply flow which also happens to be convolution
        flow = ref_layer.flow
        cfg = tf.keras.layers.serialize(flow)
        cfg['name'] = flow.name
        weights = OrderedDict()
        for k, w in wlist:
            if flow.name in k:
                weights[k] = w
        node = LayerNode(cfg, weights=flow.weights,
                         input_layers=[node],
                         metadata={'quantize_config': None})

        # Apply scale which is sort of stupid but I guess needed
        scale = ref_layer.scale
        scale_layer = tf.keras.layers.Lambda(
            lambda x: x * scale)  # sadness! but needed I guess
        cfg = tf.keras.layers.serialize(scale_layer)
        cfg['name'] = scale_layer.name
        scale_layer_node = LayerNode(
            cfg,
            weights=None,
            input_layers=[node],
            metadata={
                'quantize_config': None})

        return scale_layer_node


class FlowQuantize(transforms.Transform):
    def pattern(self):
        return LayerPattern('Flow', inputs=[LayerPattern('DownConv')])

    def replacement(self, match_layer):
        if _has_custom_quantize_config(match_layer):
            return match_layer
        """
        prv, nxt = inputs
        cost = self.cost_volume((prv, nxt))
        feat = [cost, prv, nxt]
        feat = tf.concat(feat, axis=self.axis)
        return self.flow(feat)
        """
        print('match_layer = {}'.format(match_layer))
        print('match_layer = {}'.format(match_layer.layer))
        print('inputs = {}'.format(match_layer.input_layers))  # empty?
        # reference flow layer
        ref_flow = Flow.from_config(match_layer.layer['config'])
        # Create cvol layer consistent with config.
        cvol = ref_flow.cost_volume
        cfg = tf.keras.layers.serialize(cvol)
        cfg['name'] = cvol.name
        node = LayerNode(cfg, weights=None,
                         input_layers=match_layer.input_layers,
                         metadata={'quantize_config': None})

        # NOTE(ycho): (mis)appropriating CostVolume*.axis here
        layer = tf.keras.layers.Concatenate(axis=ref_flow.axis)
        cfg = tf.keras.layers.serialize(layer)
        cfg['name'] = layer.name
        print('inputs = {}'.format([node] + match_layer.input_layers))
        node = LayerNode(cfg, weights=None,
                         input_layers=[node] + match_layer.input_layers,
                         metadata={'quantize_config': None})

        # NOTE(ycho): ONLY `flow` has weights! good news-
        # our job is made a bit easier.
        flow = ref_flow.flow
        cfg = tf.keras.layers.serialize(flow)
        cfg['name'] = flow.name
        node = LayerNode(cfg, weights=match_layer.weights,
                         input_layers=[node],
                         metadata={'quantize_config': None})
        return node


class Custom8BitQuantizeLayoutTransform(
        quantize_layout_transform.QuantizeLayoutTransform):
  """Default model transformations."""

  def apply(self, model, layer_quantize_map):
    """Implement default 8-bit transforms.

    Currently this means the following.
      1. Pull activations into layers, and apply fuse activations. (TODO)
      2. Modify range in incoming layers for Concat. (TODO)
      3. Fuse Conv2D/DepthwiseConv2D + BN into single layer.

    Args:
      model: Keras model to be quantized.
      layer_quantize_map: Map with keys as layer names, and values as dicts
        containing custom `QuantizeConfig`s which may have been passed with
        layers.

    Returns:
      (Transformed Keras model to better match TensorFlow Lite backend, updated
      layer quantize map.)
    """

    transforms = [
        default_8bit_transforms.InputLayerQuantize(),
        default_8bit_transforms.SeparableConv1DQuantize(),
        default_8bit_transforms.SeparableConvQuantize(),
        default_8bit_transforms.Conv2DReshapeBatchNormReLUQuantize(),
        default_8bit_transforms.Conv2DReshapeBatchNormActivationQuantize(),
        default_8bit_transforms.Conv2DBatchNormReLUQuantize(),
        default_8bit_transforms.Conv2DBatchNormActivationQuantize(),
        default_8bit_transforms.Conv2DReshapeBatchNormQuantize(),
        default_8bit_transforms.Conv2DBatchNormQuantize(),
        default_8bit_transforms.ConcatTransform6Inputs(),
        default_8bit_transforms.ConcatTransform5Inputs(),
        default_8bit_transforms.ConcatTransform4Inputs(),
        default_8bit_transforms.ConcatTransform3Inputs(),
        default_8bit_transforms.ConcatTransform(),
        default_8bit_transforms.AddReLUQuantize(),
        default_8bit_transforms.AddActivationQuantize(),
        OptFlowQuantize(),
        FlowQuantize()
    ]
    return model_transformer.ModelTransformer(
        model, transforms,
        set(layer_quantize_map.keys()), layer_quantize_map).transform()


class Custom8BitQuantizeScheme(quantize_scheme.QuantizeScheme):
  def get_layout_transformer(self):
    return Custom8BitQuantizeLayoutTransform()

  def get_quantize_registry(self):
    return default_8bit_quantize_registry.Default8BitQuantizeRegistry()
