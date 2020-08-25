#!/usr/bin/env python3

from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry


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
            n = self.wmap[child_attr]
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
