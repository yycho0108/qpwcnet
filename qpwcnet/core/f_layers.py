#!/usr/bin/env python3
"""
Reimplementation of stuff in `layers.py` with keras Functional API.
Let's see if this makes it easier for us to work with the final model,
in terms of preprocessing, etc.
"""

import tensorflow as tf


def down_conv_block(num_filters: int, gamma: float = 0.0004):
    """ Down-convolution, reduces spatial dimensions by half. """
    data_format = tf.keras.backend.image_data_format()

    conv_a = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=3,
        strides=2,
        activation='Mish',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(gamma),
        data_format=data_format)

    conv_aa = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=3,
        strides=1,
        activation='Mish',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(gamma),
        data_format=data_format)

    conv_b = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=3,
        strides=1,
        activation='Mish',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(gamma),
        data_format=data_format)

    return tf.keras.Sequential([conv_a, conv_aa, conv_b])
