#!/usr/bin/env python3

import itertools
from functools import wraps
from typing import Any, Callable, Optional, Sequence
import tensorflow as tf
import einops

from qpwcnet.train.robust_loss.adaptive import AdaptiveImageLossFunction, AdaptiveLossFunction


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
        # self.loss = tf.keras.losses.MeanSquaredError()
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        # if true=256, pred=8, scale=1/32
        # scale = tf.cast(y_pred.shape[1], tf.float32) / y_true.shape[1]

        if self.data_format == 'channels_first':
            numer = tf.cast(tf.shape(y_pred)[2], tf.float32)  # NCHW[2]=H
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
            numer = tf.cast(tf.shape(y_pred)[1], tf.float32)  # NHWC[1]=H
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


class FlowMseLossV2(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        self.data_format = tf.keras.backend.image_data_format()
        self.axis = _get_axis(self.data_format)
        # NOTE(ycho): In the current loss definition,
        # 0.1 amounts to a flow error magnitude that is ~10%
        # of the image space.
        self.loss = tf.keras.losses.Huber(0.1)
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        if self.data_format == 'channels_first':
            pattern = 'n c h w'
        else:
            pattern = 'n h w c'
        true_shape = einops.parse_shape(y_true, pattern)
        pred_shape = einops.parse_shape(y_pred, pattern)

        # Scale by which to multiply flow magnitude
        flow_scale = pred_shape['h'] / true_shape['h']

        # Scale by which to multiply loss magnitude
        loss_scale = 2.0 / (pred_shape['w'] + pred_shape['h'])

        # NOTE(ycho): In general, will be integer multiples,
        # so we use einops.reduce() instead of e.g. resize_bilinear()
        if self.data_format == 'channels_first':
            y_true = flow_scale * einops.reduce(y_true,
                                                'n c (h sh) (w sw) -> n c h w',
                                                'mean', sh=true_shape['h'] //
                                                pred_shape['h'],
                                                sw=true_shape['w'] //
                                                pred_shape['w'])
        else:
            y_true = flow_scale * einops.reduce(y_true,
                                                'n (h sh) (w sw) c -> n h w c',
                                                'mean', sh=true_shape['h'] //
                                                pred_shape['h'],
                                                sw=true_shape['w'] //
                                                pred_shape['w'])
        return self.loss(loss_scale * y_true, loss_scale * y_pred)


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


class AdaptiveResizedLoss(tf.keras.losses.Loss):
    def __init__(self, shape, *args, **kwargs):
        self._config = {
            'shape': shape
        }
        self.data_format = tf.keras.backend.image_data_format()

        if self.data_format == 'channels_first':
            #NCHW
            img_shape = [shape[2], shape[3], shape[1]]
        else:
            #NHWC
            img_shape = [shape[1], shape[2], shape[3]]

        # NOTE(ycho): `2` here indicates that
        # This class is specialized for computing loss on flow.
        self.loss = AdaptiveImageLossFunction(img_shape, tf.float32,
                                              color_space='RGB',
                                              representation='DCT',  # HMM...
                                              )
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        if self.data_format == 'channels_first':
            pattern = 'n c h w'
        else:
            pattern = 'n h w c'
        true_shape = einops.parse_shape(y_true, pattern)
        pred_shape = einops.parse_shape(y_pred, pattern)

        # Scale by which to multiply flow magnitude
        flow_scale = pred_shape['h'] / true_shape['h']

        # Scale by which to multiply loss magnitude
        loss_scale = 1.0 / (pred_shape['w'] * pred_shape['h'])

        # NOTE(ycho): In general, will be integer multiples,
        # so we use einops.reduce() instead of e.g. resize_bilinear()
        if self.data_format == 'channels_first':
            y_pred = einops.rearrange(y_pred, 'n c h w -> n h w c')
            y_true = flow_scale * einops.reduce(y_true,
                                                'n c (h sh) (w sw) -> n h w c',
                                                'mean', sh=true_shape['h'] //
                                                pred_shape['h'],
                                                sw=true_shape['w'] //
                                                pred_shape['w'])
        else:
            y_true = flow_scale * einops.reduce(y_true,
                                                'n (h sh) (w sw) c -> n h w c',
                                                'mean', sh=true_shape['h'] //
                                                pred_shape['h'],
                                                sw=true_shape['w'] //
                                                pred_shape['w'])

        # finally, we call the loss...
        loss = self.loss(loss_scale * (y_true - y_pred))
        return loss

    def get_config(self):
        config = super().get_config().copy()
        config.update(self._config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def extend_and_filter(
    extend_method: Callable[..., Sequence], filter_method: Optional[Callable[..., Sequence]] = None,
) -> Callable[[Any], Any]:
    """
    This decorator calls a decorated method, and extends the result with another method
    on the same class. This method is called after the decorated function, with the same
    arguments as the decorated function. If specified, a second filter method can be applied
    to the extended list. Filter method should also be a method from the class.

    :param extend_method: Callable
        Accepts the same argument as the decorated method.
        The returned list from `extend_method` will be added to the
        decorated method's returned list.
    :param filter_method: Callable
        Takes in the extended list and filters it.
        Defaults to no filtering for `filter_method` equal to `None`.
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapped(self, *args, **kwargs):  # type: ignore
            ret = f(self, *args, **kwargs)
            ret.extend(extend_method(self, *args, **kwargs))
            ret = filter_method(
                self, ret) if filter_method is not None else ret
            return ret

        return wrapped

    return decorator


class AdaptiveResizedLossV2(tf.keras.losses.Loss):
    def __init__(self, loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.data_format = tf.keras.backend.image_data_format()
        # NOTE(ycho): `2` here indicates that
        # This class is specialized for computing loss on flow.

    def call(self, y_true, y_pred):
        if self.data_format == 'channels_first':
            pattern = '_ c h w'
        else:
            pattern = '_ h w c'
        true_shape = einops.parse_shape(y_true, pattern)
        pred_shape = einops.parse_shape(y_pred, pattern)

        # Scale by which to multiply flow magnitude
        flow_scale = pred_shape['h'] / true_shape['h']

        # Scale by which to multiply loss magnitude
        loss_scale = 1.0 / (pred_shape['w'] * pred_shape['h'])

        # NOTE(ycho): In general, will be integer multiples,
        # so we use einops.reduce() instead of e.g. resize_bilinear()
        if self.data_format == 'channels_first':
            y_true = flow_scale * einops.reduce(y_true,
                                                'n c (h sh) (w sw) -> n c h w',
                                                'mean', sh=true_shape['h'] //
                                                pred_shape['h'],
                                                sw=true_shape['w'] //
                                                pred_shape['w'])
        else:
            y_true = flow_scale * einops.reduce(y_true,
                                                'n (h sh) (w sw) c -> n h w c',
                                                'mean', sh=true_shape['h'] //
                                                pred_shape['h'],
                                                sw=true_shape['w'] //
                                                pred_shape['w'])

        # Finally, we call the loss...
        raw_loss = loss_scale * (y_true - y_pred)

        # NOTE(ycho): Treat ALF loss distribution over the flow channels.
        if self.data_format == 'channels_first':
            raw_loss = einops.rearrange(raw_loss, 'n c h w -> (n h w) c')
        else:
            raw_loss = einops.rearrange(raw_loss, 'n h w c -> (n h w) c')
        loss = tf.reduce_mean(self.loss_func(raw_loss))
        return loss
