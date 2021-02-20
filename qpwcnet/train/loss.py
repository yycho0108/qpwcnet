#!/usr/bin/env python3

import tensorflow as tf


def _get_axis(data_format: str):
    """ Get channel axis from data format. """
    axis = None
    if data_format == 'channels_first':
        axis = 1
    elif data_format == 'channels_last':
        axis = 3
    else:
        raise ValueError(
            'Unsupported data format : {}'.format(data_format))
    return axis


class FlowMseLoss(tf.keras.losses.Loss):
    def __init__(self, data_format='channels_first', *args, **kwargs):
        self._config = {
            'data_format': data_format
        }
        self.data_format = data_format
        self.axis = _get_axis(data_format)
        # self.scale = tf.constant(1.0 / scale)
        # self.weight = tf.constant(weight)
        # self.mse = tf.keras.losses.MeanSquaredError()
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        # if true=256, pred=8, scale=1/32
        # scale = tf.cast(y_pred.shape[1], tf.float32) / y_true.shape[1]

        if self.data_format == 'channels_first':
            numer = tf.cast(tf.shape(y_pred)[2], tf.float32)
            denom = tf.cast(tf.shape(y_true)[2], tf.float32)
            scale = numer / denom

            y_true_nhwc = tf.transpose(y_true, (0, 2, 3, 1))
            y_true_down_nhwc = tf.image.resize(
                y_true_nhwc, tf.shape(y_pred)[2:4]) * scale
            y_true_down = tf.transpose(y_true_down_nhwc, (0, 3, 1, 2))
            # NOTE(yycho0108): 0.05 here effectively scales flow-related loss by 1/20x.
            # err = tf.norm(0.05*(y_true_down - y_pred), ord=2, axis=self.axis)
            # NOTE(yycho0108): delta=4.0 since search_range==4
            loss = tf.reduce_mean(
                tf.norm(y_true_down - y_pred, ord=2, axis=self.axis))
            return loss
        elif self.data_format == 'channels_last':
            numer = tf.cast(tf.shape(y_pred)[1], tf.float32)
            denom = tf.cast(tf.shape(y_true)[1], tf.float32)
            scale = numer / denom

            y_true_down = tf.image.resize(
                y_true, tf.shape(y_pred)[1:3]) * scale
            # loss = tf.reduce_mean(tf.losses.huber(
            #    y_true_down, y_pred, delta=4.0))
            loss = tf.reduce_mean(
                tf.norm(y_true_down - y_pred, ord=2, axis=self.axis))
            return loss
            # NOTE(yycho0108): 0.05 here effectively scales flow-related loss by 1/20x.
            # err = tf.norm(0.05*(y_true_down - y_pred),
            #              ord=2, axis=self.axis)
            #loss = tf.reduce_mean(tf.reduce_sum(err, axis=(1, 2)))
            #weight = 0.0003125 / (scale * scale)
            # return weight * loss

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FlowMseLossFineTune(tf.keras.losses.Loss):
    def __init__(self, data_format='channels_first',
                 q=0.4, eps=0.01, *args, **kwargs):
        self.data_format = data_format
        self.axis = _get_axis(data_format)
        self.q = q
        self.eps = eps
        self.config_ = {
            'data_format': data_format,
            'q': q,
            'eps': eps
        }
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        # if true=256, pred=8, scale=1/32
        # scale = tf.cast(y_pred.shape[1], tf.float32) / y_true.shape[1]

        if self.data_format == 'channels_first':
            numer = tf.cast(tf.shape(y_pred)[2], tf.float32)
            denom = tf.cast(tf.shape(y_true)[2], tf.float32)
            scale = numer / denom

            y_true_nhwc = tf.transpose(y_true, (0, 2, 3, 1))
            y_true_down_nhwc = tf.image.resize(
                y_true_nhwc, tf.shape(y_pred)[2:4]) * scale
            y_true_down = tf.transpose(y_true_down_nhwc, (0, 3, 1, 2))
        elif self.data_format == 'channels_last':
            numer = tf.cast(tf.shape(y_pred)[1], tf.float32)
            denom = tf.cast(tf.shape(y_true)[1], tf.float32)
            scale = numer / denom

            y_true_down = tf.image.resize(
                y_true, tf.shape(y_pred)[1:3]) * scale
        err_norm = tf.norm(y_true_down - y_pred, ord=1, axis=self.axis)
        err_norm = tf.pow(err_norm + self.eps, self.q)
        loss = tf.reduce_mean(err_norm)
        return loss  # scale * 10.0

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AutoResizeMseLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.data_format = data_format
        self.axis = _get_axis(data_format)
        self.loss = tf.keras.losses.MeanSquaredError()
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        if self.data_format == 'channels_first':
            # both y_true and y_pred are NCHW
            y_true_nhwc = tf.transpose(y_true, (0, 2, 3, 1))
            y_true_down_nhwc = tf.image.resize(
                y_true_nhwc, tf.shape(y_pred)[2:4])
            y_true_down = tf.transpose(y_true_down_nhwc, (0, 3, 1, 2))
            loss = self.loss(y_true_down, y_pred)
            return loss
        elif self.data_format == 'channels_last':
            y_true_down = tf.image.resize(y_true, tf.shape(y_pred)[1:3])
            loss = self.loss(y_true_down, y_pred)
            return loss
