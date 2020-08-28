#!/usr/bin/env python3

import tensorflow as tf


class FlowMseLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        # self.scale = tf.constant(1.0 / scale)
        # self.weight = tf.constant(weight)
        # self.mse = tf.keras.losses.MeanSquaredError()
        super().__init__(*args, **kwargs)

    def build(self, input_shapes):
        print('fmseloss', input_shapes)
        super().build(input_shapes)

    def call(self, y_true, y_pred):
        # if true=256, pred=8, scale=1/32
        # scale = tf.cast(y_pred.shape[1], tf.float32) / y_true.shape[1]

        numer = tf.cast(tf.shape(y_pred)[1], tf.float32)
        denom = tf.cast(tf.shape(y_true)[1], tf.float32)
        scale = numer / denom

        y_true_down = tf.image.resize(y_true, tf.shape(y_pred)[1:3]) * scale
        # NOTE(yycho0108): 0.05 here effectively scales flow-related loss by 1/20x.
        err_norm = tf.norm(0.05*(y_true_down - y_pred), ord=2, axis=3)
        loss = tf.reduce_mean(tf.reduce_sum(err_norm, axis=(1, 2)))
        weight = 0.0003125 / (scale * scale)
        return weight * loss

        #weight = 0.32 * scale
        # return weight * tf.reduce_mean(tf.square(y_true_down - y_pred))


class FlowMseLossFineTune(tf.keras.losses.Loss):
    def __init__(self, q=0.4, eps=0.01, *args, **kwargs):
        self.q = q
        self.eps = eps
        self.config_ = {
            'q': q,
            'eps': eps
        }
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        # if true=256, pred=8, scale=1/32
        # scale = tf.cast(y_pred.shape[1], tf.float32) / y_true.shape[1]

        numer = tf.cast(tf.shape(y_pred)[1], tf.float32)
        denom = tf.cast(tf.shape(y_true)[1], tf.float32)
        scale = numer / denom

        y_true_down = tf.image.resize(y_true, tf.shape(y_pred)[1:3]) * scale
        err_norm = tf.norm(y_true_down - y_pred, ord=1, axis=3)
        err_norm = tf.pow(err_norm + self.eps, self.q)
        loss = tf.reduce_mean(tf.reduce_sum(err_norm, axis=(1, 2)))
        weight = 0.0003125 / (scale * scale)
        return weight * loss  # tf.pow(loss, 2)

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
