#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import Default8BitOutputQuantizeConfig


# class ConvBN(tf.keras.layers.Layer):
class ConvBN(tf.Module):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.conv = tf.keras.layers.Conv2D(16, 3, padding='valid')
        self.norm = tf.keras.layers.BatchNormalization(name=self.name + 'bn')
        self.conv2 = tf.keras.layers.Conv2D(16, 3, padding='valid')

    def __call__(self, x):
        # return self.conv2(self.norm(self.conv(x)))
        return 5.0 + self.norm(self.conv(x))


def build_model(data_format: str = None):
    """ build simple u-net model. """
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_first':
        axis = -3
        x1 = tf.keras.layers.Input(shape=(3, 256, 512))
        x2 = tf.keras.layers.Input(shape=(3, 256, 512))
    else:
        axis = -1
        x1 = tf.keras.layers.Input(shape=(256, 512, 3))
        x2 = tf.keras.layers.Input(shape=(256, 512, 3))
    convbn = ConvBN(name='cb')
    b1 = convbn(x1)
    b2 = convbn(x2)

    convbn = ConvBN(name='cb2')
    b1 = convbn(b1)
    b2 = convbn(b2)
    cat = tf.keras.layers.Concatenate(axis=axis)
    out = cat((b1, b2))
    return tf.keras.Model(inputs=(x1, x2), outputs=out)


def quantize_annotate_layer(layer: tf.keras.layers.Layer):
    q = tfmot.quantization.keras.quantize_annotate_layer
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        return q(layer,
                 # tfmot.quantization.keras.default_8bit.  Default8BitQuantizeRegistry().get_quantize_config(layer)
                 Default8BitOutputQuantizeConfig()
                 )

    if isinstance(layer, tf.keras.layers.Concatenate):
        return layer
    if isinstance(layer, tf.keras.layers.Conv2D):
        return q(layer)
    return q(layer)


def quantize_model(model: tf.keras.Model):
    with tfmot.quantization.keras.quantize_scope({
        'Default8BitOutputQuantizeConfig': Default8BitOutputQuantizeConfig,
        'ConvBN': ConvBN
    }):

        print(model.to_json())

        print('<annotate>')
        annotated_model = tf.keras.models.clone_model(
            model,
            clone_function=quantize_annotate_layer)
        print('</annotate>')

        print('<quantize>')
        quantized_model = tfmot.quantization.keras.quantize_apply(
            annotated_model)
        print('</quantize>')

        #print('<compile>')
        #quantized_model.compile(optimizer='adam')  # hmm
        #print('</compile>')

        quantized_model.summary()
        return quantized_model


def main():
    model = build_model()
    print(model.summary())
    quantized_model = quantize_model(model)


if __name__ == '__main__':
    main()
