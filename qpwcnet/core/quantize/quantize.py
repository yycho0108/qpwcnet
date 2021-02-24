#!/usr/bin/env python3

import inspect
import tensorflow as tf
from functools import partial
from collections import OrderedDict
from typing import Callable

from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import Default8BitOutputQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras import quantize_scheme
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry

from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms
LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern


#class OptFlowXfm(transforms.Transform):
#  """Conv2DBatchNormFold."""
#
#  def pattern(self):
#    return LayerPattern('OptFlow', {}, None)
#
#  def replacement(self, match_layer):
#      match_layer._flatten_layers(/
#    conv_layer, bn_layer = _get_conv_bn_layers(match_layer)
#
#    fused_params = _get_params(conv_layer, bn_layer)
#    fused_layer = _ConvBatchNorm2D(**fused_params)
#
#    weights = _get_weights(match_layer)
#    return _get_layer_node(fused_layer, weights)
#
#  def custom_objects(self):
#    return {'_ConvBatchNorm2D': _ConvBatchNorm2D}


#class CustomTransform(
#        quantize_layout_transform.QuantizeLayoutTransform):
#  """Default model transformations."""
#
#  def apply(self, model, layer_quantize_map):
#    """Implement default 8-bit transforms.
#
#    Currently this means the following.
#      1. Pull activations into layers, and apply fuse activations. (TODO)
#      2. Modify range in incoming layers for Concat. (TODO)
#      3. Fuse Conv2D/DepthwiseConv2D + BN into single layer.
#
#    Args:
#      model: Keras model to be quantized.
#      layer_quantize_map: Map with keys as layer names, and values as dicts
#        containing custom `QuantizeConfig`s which may have been passed with
#        layers.
#
#    Returns:
#      (Transformed Keras model to better match TensorFlow Lite backend, updated
#      layer quantize map.)
#    """
#
#    transforms = [
#        default_8bit_transforms.InputLayerQuantize(),
#        default_8bit_transforms.SeparableConv1DQuantize(),
#        default_8bit_transforms.SeparableConvQuantize(),
#        default_8bit_transforms.Conv2DReshapeBatchNormReLUQuantize(),
#        default_8bit_transforms.Conv2DReshapeBatchNormActivationQuantize(),
#        default_8bit_transforms.Conv2DBatchNormReLUQuantize(),
#        default_8bit_transforms.Conv2DBatchNormActivationQuantize(),
#        default_8bit_transforms.Conv2DReshapeBatchNormQuantize(),
#        default_8bit_transforms.Conv2DBatchNormQuantize(),
#        default_8bit_transforms.ConcatTransform6Inputs(),
#        default_8bit_transforms.ConcatTransform5Inputs(),
#        default_8bit_transforms.ConcatTransform4Inputs(),
#        default_8bit_transforms.ConcatTransform3Inputs(),
#        default_8bit_transforms.ConcatTransform(),
#        default_8bit_transforms.AddReLUQuantize(),
#        default_8bit_transforms.AddActivationQuantize(),
#    ]
#    return model_transformer.ModelTransformer(
#        model, transforms,
#        set(layer_quantize_map.keys()), layer_quantize_map).transform()
#
#
#class CustomQuantizeScheme(quantize_scheme.QuantizeScheme):
#  def get_layout_transformer(self):
#    return CustomTransform()
#
#  def get_quantize_registry(self):
#    return default_8bit_quantize_registry.Default8BitQuantizeRegistry()

#def deep_quantize_annotate_layer(
#        layer,
#        anno_fn: Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer],
#        is_leaf: Callable[[tf.keras.layers.Layer], bool]):
#    """
#    quantize_annotate_layer() analogue that works on nested Layers.
#    """
#    sub_layers = inspect.getmembers(
#        layer, lambda attr: isinstance(
#            attr, tf.keras.layers.Layer))
#    out = anno_fn(layer)
#    if not is_leaf(layer):
#        for name, sub_layer in sub_layers:
#            new_layer = deep_quantize_annotate_layer(sub_layer, anno_fn)
#            # Would setattr work? Or do we need to pull other tricks?
#            setattr(out, name, new_layer)
#    return out
#
#


def replace_layer(model,
                  layer_predicate: Callable[[tf.keras.layers.Layer], bool],
                  layer_factory: Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer],
                  insert_layer_name=None):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    def _build_network(layer: tf.keras.layers.Layer):
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the input layers of each layer, etc etc
    for layer in model.layers:
        layers = layer._flatten_layers(recursive=True, include_self=True)
        for l in layers:
            _build_network(l)

    # Set the output tensor of the input layer
    # NOTE(ycho): would still ~work even with the `include_self=True`
    # scheme since `self` is returned first.
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    def _process_layer(layer: tf.keras.layers.Layer):
        # Determine input tensors
        layer_input = [
            network_dict['new_output_tensor_of'][layer_aux]
            for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if layer_predicate(layer):
            x = layer_input

            new_layer = layer_factory(layer, insert_layer_name)

            # Determine name ...
            # if insert_layer_name:
            #     new_layer.name = insert_layer_name
            # else:
            #     print('new name = {}_{}'.format(layer.name,
            #                                     new_layer.name))
            #     new_layer.name = '{}_{}'.format(layer.name,
            #                                     new_layer.name)

            # FIXME(ycho): Would this work for layer.call() with >1 args?
            x = new_layer(x)
            print('New layer: {} Old layer: {}'.format(
                new_layer.name, layer.name))
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    # Iterate over all layers after the input.
    model_outputs = []
    for layer in model.layers[1:]:
        layers = layer._flatten_layers(recursive=True, include_self=True)
        for l in layers:
            _process_layer(l)

    return Model(inputs=model.inputs, outputs=model_outputs)


"""
def flatten_model(model: tf.keras.Model):
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    def _build_network(layer: tf.keras.layers.Layer):
        # Add self
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

        # Add sub-layers
        sub_layers = inspect.getmembers(
            layer, lambda attr: isinstance(
                attr, tf.keras.layers.Layer))
        for sub_layer in sub_layers:
            _build_network(sub_layer)

    for layer in model.layers:
        _build_network(layer)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:
        # Determine input tensors
        layer_input = [
            network_dict['new_output_tensor_of'][layer_aux]
            for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        x = layer_input
        new_layer = insert_layer_factory()
        if insert_layer_name:
            new_layer.name = insert_layer_name
        else:
            new_layer.name = '{}_{}'.format(layer.name,
                                            new_layer.name)
        x = new_layer(x)
        print('New layer: {} Old layer: {}'.format(
            new_layer.name, layer.name, position))

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)
"""


class RecursiveDelegateConfig(QuantizeConfig):
    def __init__(self, registry=None):
        if registry is None:
            registry = default_8bit_quantize_registry.Default8BitQuantizeRegistry()
        self.registry = registry
        self.wmap = OrderedDict()
        self.amap = OrderedDict()

    @staticmethod
    def get_sub_layers(layer: tf.keras.layers.Layer):
        layers = layer._flatten_layers(recursive=False, include_self=False)
        return sorted([(l.name, l) for l in layers])

    def get_weights_and_quantizers(self, layer):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            config = Default8BitOutputQuantizeConfig()
            out = config.get_weights_and_quantizers(layer)
            print('get/ for {}:{} -> {}'.format(layer, layer.name,
                                                [w.shape for w, q in out]))
            return out

        # First, try if supported by default.
        if self.registry.supports(layer):
            config = self.registry.get_quantize_config(layer)
            out = config.get_weights_and_quantizers(layer)
            print('get/ for {}:{} -> {}'.format(layer, layer.name,
                                                [w.shape for w, q in out]))
            return out
        else:
            # Separable Conv, Warp, Mish, Cost Volume
            print('Unsupported layer : {}:{}'.format(layer, layer.name))

        # Otherwise, assume this is a compositional layer
        # and process recursively. The requirement here is that
        # all leaf-node layers must be supported. Also,
        # `self` cannot have non-child weights -- just to preserve sanity.
        out = []
        sub_layers = self.get_sub_layers(layer)
        for name, sub_layer in sub_layers:
            # NOTE(ycho): Might want to dedup_weights
            wnq = self.get_weights_and_quantizers(sub_layer)
            self.wmap[name] = len(wnq)
            out.extend(wnq)
        if not out:
            print('empty output : are you sure? {}'.format(layer))
        return out

    def get_activations_and_quantizers(self, layer):
        # First, try if supported by default.
        if self.registry.supports(layer):
            config = self.registry.get_quantize_config(layer)
            return config.get_activations_and_quantizers(layer)

        out = []
        sub_layers = self.get_sub_layers(layer)
        for name, sub_layer in sub_layers:
            anq = self.get_activations_and_quantizers(sub_layer)
            self.amap[name] = len(anq)
            out.extend(anq)
        return out

    def set_quantize_weights(self, layer, quantize_weights):
        # NOTE(ycho): Alternatively, use a flattened index from
        # (presumably) previous calls to corresponding get_* funcs.

        if isinstance(layer, tf.keras.layers.BatchNormalization):
            config = Default8BitOutputQuantizeConfig()
            out = config.set_quantize_weights(layer, quantize_weights)
            print('set/ for {} -> {}'.format(layer,
                                             [w.shape for w in quantize_weights]))
            return out

        if self.registry.supports(layer):
            print('set/ for {} -> {}'.format(layer,
                                             [w.shape for w in quantize_weights]))
            config = self.registry.get_quantize_config(layer)
            return config.set_quantize_weights(layer, quantize_weights)

        sub_layers = self.get_sub_layers(layer)
        for name, sub_layer in sub_layers:
            n = self.wmap[name]
            self.set_quantize_weights(sub_layer, quantize_weights[:n])
            quantize_weights = quantize_weights[n:]

    def set_quantize_activations(self, layer, quantize_activations):
        # First, try if supported by default.
        if self.registry.supports(layer):
            config = self.registry.get_quantize_config(layer)
            return config.set_quantize_activations(layer, quantize_activations)

        sub_layers = self.get_sub_layers(layer)
        for name, sub_layer in sub_layers:
            n = self.amap[name]
            self.set_quantize_activations(sub_layer, quantize_activations[:n])
            quantize_activations = quantize_activations[n:]

    def get_output_quantizers(self, layer):
        # First, try if supported by default.
        if self.registry.supports(layer):
            config = self.registry.get_quantize_config(layer)
            return config.get_output_quantizers(layer)

        sub_layers = self.get_sub_layers(layer)
        out = []
        for name, sub_layer in sub_layers:
            out.extend(self.get_output_quantizers(sub_layer))

        return out

    @classmethod
    def from_config(cls, config):
        """Instantiates a `Default8BitQuantizeConfig` from its config.

        Args:
            config: Output of `get_config()`.

        Returns:
            A `Default8BitQuantizeConfig` instance.
        """
        return cls(**config)

    def get_config(self):
        return {
            'registry': self.registry,
        }

    def __eq__(self, other):
        return isinstance(
            other, RecursiveDelegateConfig) and (
            self.get_config() == other.get_config())

    def __ne__(self, other):
        return not self.__eq__(other)


class DelegateConvConfig(QuantizeConfig):
    def __init__(self, child_attrs):
        self.child_attrs = child_attrs
        self.child_config = default_8bit_quantize_registry.Default8BitConvQuantizeConfig(
            ['kernel'], ['activation'], False)

        self.wmap = {}
        self.amap = {}

    def get_weights_and_quantizers(self, layer):
        out = []
        for child_attr in self.child_attrs:
            child = getattr(layer, child_attr)
            wq = self.child_config.get_weights_and_quantizers(child)
            self.wmap[child_attr] = len(wq)
            out.extend(wq)
        return out

    def get_activations_and_quantizers(self, layer):
        out = []
        for child_attr in self.child_attrs:
            child = getattr(layer, child_attr)
            wa = self.child_config.get_activations_and_quantizers(child)
            self.amap[child_attr] = len(wa)
            out.extend(wa)
        return out

    def set_quantize_weights(self, layer, quantize_weights):
        for child_attr in self.child_attrs:
            child = getattr(layer, child_attr)
            n = self.wmap[child_attr]
            self.child_config.set_quantize_weights(child, quantize_weights[:n])
            quantize_weights = quantize_weights[n:]

    def set_quantize_activations(self, layer, quantize_activations):
        for i, child_attr in enumerate(self.child_attrs):
            child = getattr(layer, child_attr)
            n = self.amap[child_attr]
            self.child_config.set_quantize_activations(
                child, quantize_activations[:n])
            quantize_activations = quantize_activations[n:]

    def get_output_quantizers(self, layer):
        out = []
        for i, child_attr in enumerate(self.child_attrs):
            child = getattr(layer, child_attr)
            out.extend(self.child_config.get_output_quantizers(child))
        return out

    @classmethod
    def from_config(cls, config):
        """Instantiates a `Default8BitQuantizeConfig` from its config.

        Args:
            config: Output of `get_config()`.

        Returns:
            A `Default8BitQuantizeConfig` instance.
        """
        return cls(**config)

    def get_config(self):
        # TODO(pulkitb): Add weight and activation quantizer to config.
        # Currently it's created internally, but ideally the quantizers should be
        # part of the constructor and passed in from the registry.
        return {
            'child_attrs': self.child_attrs,
        }

    def __eq__(self, other):
        if not isinstance(other, DelegateConvConfig):
            return False

        return (self.child_attrs == other.child_attrs and
                self.child_config == other.child_config)

    def __ne__(self, other):
        return not self.__eq__(other)

#def deep_quantize_apply(
#    model: tf.keras.Model,
#        scheme=default_8bit_quantize_scheme.Default8BitQuantizeScheme()):
#
#  if model is None:
#    raise ValueError('`model` cannot be None')
#
#  if not isinstance(model, keras.Model):
#    raise ValueError('`model` can only be a `tf.keras.Model` instance.'
#                     'You passed an instance of type: {input}.'.format(
#                         input=model.__class__.__name__))
#
#  if not isinstance(model, keras.Sequential) \
#          and not model._is_graph_network:  # pylint: disable=protected-access
#    raise ValueError('`model` can only either be a tf.keras Sequential or '
#                     'Functional model.')
#
#  # Have at least 1 layer annotated with QuantizeAnnotate
#  #if not any(isinstance(layer, quantize_annotate_mod.QuantizeAnnotate)
#  #           for layer in model.layers):
#  #  raise ValueError(
#  #      '`model` must contain at least one layer which have been '
#  #      'annotated with `quantize_annotate*`. There are no layers '
#  #      'to quantize.')
#
#  if not model.built:
#    raise ValueError('`model` must be a built model. '
#                     'been built yet. Please call `model.build(input_shape)` '
#                     'before quantizing your model.')
#
#  def _clone_model_with_weights(model_to_clone):
#    cloned_model = keras.models.clone_model(model_to_clone)
#    cloned_model.set_weights(model_to_clone.get_weights())
#
#    return cloned_model
#
#  def _extract_original_model(model_to_unwrap):
#    """Extracts original model by removing wrappers."""
#    layer_quantize_map = {}
#
#    def _unwrap(layer):
#
#      # NOTE(ycho): For appending to layer_quantize_map.
#      sub_layers = inspect.getmembers(
#          layer, lambda attr: isinstance(
#              attr, tf.keras.layers.Layer))
#      for l in sub_layers:
#          _unwrap(l)
#
#      if not isinstance(layer, quantize_annotate_mod.QuantizeAnnotate):
#        return layer
#
#      annotate_wrapper = layer
#      layer_quantize_map[annotate_wrapper.layer.name] = {
#          'quantize_config': annotate_wrapper.quantize_config
#      }
#      return annotate_wrapper.layer
#
#    unwrapped_model = keras.models.clone_model(
#        model_to_unwrap, input_tensors=None, clone_function=_unwrap)
#
#    return unwrapped_model, layer_quantize_map
#
#  def _quantize(layer):  # pylint: disable=missing-docstring
#    if layer.name not in layer_quantize_map:
#      return layer
#
#    quantize_config = layer_quantize_map[layer.name].get('quantize_config')
#    if not quantize_config and quantize_registry.supports(layer):
#      quantize_config = quantize_registry.get_quantize_config(layer)
#
#    if not quantize_config:
#      error_msg = (
#          'Layer {}:{} is not supported. You can quantize this '
#          'layer by passing a `tfmot.quantization.keras.QuantizeConfig` '
#          'instance to the `quantize_annotate_layer` '
#          'API.')
#      raise RuntimeError(
#          error_msg.format(layer.name, layer.__class__,
#                           quantize_registry.__class__))
#
#    # `QuantizeWrapper` does not copy any additional layer params from
#    # `QuantizeAnnotate`. This should generally be fine, but occasionally
#    # `QuantizeAnnotate` wrapper may contain `batch_input_shape` like params.
#    # TODO(pulkitb): Ensure this does not affect model cloning.
#    return quantize_wrapper.QuantizeWrapper(layer, quantize_config)
#
#  # 1. Create a copy of the model with the same weights. This ensures
#  # modifications don't affect the original model, or its weights.
#  try:
#    model_copy = _clone_model_with_weights(model)
#  except ValueError:
#    raise ValueError(
#        'Unable to clone model. This generally happens if you used custom Keras layers or objects '
#        'in your model. Please specify them via `quantize_scope` for your calls to `quantize_model` '
#        'and `quantize_apply`.')
#
#  # 2. Remove QuantizeAnnotate wrappers from the layers in the model. This
#  # extracts the original model structure (easier to transform), and
#  # stores relevant quantization information in a map.
#  unwrapped_model, layer_quantize_map = _extract_original_model(model_copy)
#  # Model cloning excludes input layers. Add input layers into the map
#  # since they need to be matched for patterns as well.
#  # pylint: disable=protected-access
#  for input_layer in unwrapped_model._input_layers:
#    for outbound_node in input_layer._outbound_nodes:
#      if outbound_node.outbound_layer.name in layer_quantize_map:
#        layer_quantize_map[input_layer.name] = {}
#  # pylint: enable=protected-access
#
#  # 3. Apply the graph transformations required to match model passes on
#  # target device/dialect.
#  quantize_transform = scheme.get_layout_transformer()
#  # layer_quantize_map gets modified by the transformations.
#  transformed_model, layer_quantize_map = quantize_transform.apply(
#      unwrapped_model, layer_quantize_map)
#
#  # TODO(pulkitb): Think more about how to introduce Default specific code.
#  quantize_registry = scheme.get_quantize_registry()
#
#  # 4. Actually quantize all the relevant layers in the model. This is done by
#  # wrapping the layers with QuantizeWrapper, and passing the associated
#  # `QuantizeConfig`.
#
#  return keras.models.clone_model(
#      transformed_model, input_tensors=None, clone_function=_quantize)
