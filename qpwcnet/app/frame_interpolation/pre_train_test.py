#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_addons.layers.optical_flow
import einops
from pathlib import Path
import logging
from dataclasses import dataclass
from simple_parsing import Serializable
from typing import Tuple
import multiprocessing as mp

from qpwcnet.core.pwcnet import build_interpolator
from qpwcnet.core.vis import flow_to_image, cost_volume_to_flow
from qpwcnet.core.warp import tf_warp

from qpwcnet.data.youtube_vos import (
    YoutubeVosTriplet,
    YoutubeVosSettings,
    YoutubeVosTripletSettings)
from qpwcnet.data.vimeo_triplet import (
    VimeoTriplet,
    VimeoTripletSettings)
from qpwcnet.data.triplet_dataset_ops import read_triplet_dataset

from qpwcnet.train.loss import AutoResizeMseLoss
from qpwcnet.train.util import load_weights
from qpwcnet.vis.show import show

from qpwcnet.app.util.arg_setup import with_args


@dataclass
class Settings(Serializable):
    data_format: str = 'channels_first'
    input_shape: Tuple[int, int] = (256, 512)
    model: str = ''
    dataset: str = 'vimeo'
    log_level: str = 'info'


def _show(key, img, data_format: str = None, export: bool = True):
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    # Need to convert->numpy, due to tf.Tensor not supporting assignment.
    if tf.is_tensor(img) and tf.executing_eagerly():
        img = img.numpy()

    # NOTE(ycho): In order to figure out if we have aligned the frames correctly,
    # rather than just copying the inputs, draw a grid on top of the image.
    if data_format == 'channels_first':
        img[..., :, :, ::32] = 1.0
        img[..., :, ::32, :] = 1.0
    else:
        img[..., :, ::32, :] = 1.0
        img[..., ::32, :, :] = 1.0

    # FIXME(ycho): Temporary hack for exporting data.
    if export:
        tmp = img.copy()

        # clip to range

        # float32->uint8
        if issubclass(tmp.dtype.type, np.floating):
            tmp = np.clip(tmp, 0.0, 1.0)
            tmp = (255 * tmp).astype(np.uint8)

        # chw->hwc
        if data_format == 'channels_first':
            tmp = np.transpose(tmp, (1, 2, 0))

        # rgb->bgr, write
        cv2.imwrite('/tmp/{}.png'.format(key), tmp[..., ::-1])

    return show(key, img, True, data_format)


@with_args(Settings)
def main(args: Settings):
    data_format = args.data_format
    tf.keras.backend.set_image_data_format(data_format)
    model_file = Path(args.model)

    multi_output = True

    # Define inference-only model.
    model = build_interpolator(
        input_shape=args.input_shape,
        output_multiscale=False)
    load_weights(model, args.model)
    multi_output = False

    logging.info('Done with model load')

    # Extract flow-only model for visualization.
    # NOTE(ycho): We're only extracting forward-directional flow,
    # i.e. flow : prv[i, j] == nxt[i+flo[i,j,1], j+flo[i,j,0]]
    flow_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('lambda_11').get_output_at(0)
        # print(model.get_layer('lambda_11').get_output_at(1))
    )

    # FIXME(ycho): Ability to select dataset
    # Select dataset.
    if args.dataset == 'ytvos':
        dataset = YoutubeVos(YoutubeVosSettings(data_type='train'))
    elif args.dataset == 'vimeo':
        dataset = VimeoTriplet(VimeoTripletSettings(data_type='train'))
    else:
        raise ValueError('Invalid dataset = {}'.format(args.dataset))
    # TripletDataset -> tf.data.Dataset
    dataset = read_triplet_dataset(dataset, dsize=args.input_shape,
                                   augment=False,
                                   batch_size=1)

    for img0, img1, img2 in dataset:
        img_pair = tf.concat([img0, img2], axis=3)

        # @see pre_train:preprocess()
        if data_format == 'channels_first':
            img_pair = einops.rearrange(img_pair, 'n h w c -> n c h w')
        img_pair -= 0.5

        if True:
            flow = flow_model(img_pair)
            flow_rgb = flow_to_image(flow, data_format=data_format)
            _show('5-flow', flow_rgb[0], data_format)

            # warp 1 -> 0, let's see how it fares.
            if data_format == 'channels_first':
                upflow = 2.0 * einops.repeat(flow,
                                             'n c h w -> n c (h h2) (w w2)',
                                             h2=2, w2=2)
            else:
                upflow = 2.0 * einops.repeat(flow,
                                             'n h w c -> n (h h2) (w w2) c',
                                             h2=2, w2=2)
            if data_format == 'channels_first':
                img1_ = einops.rearrange(img1, 'n h w c -> n c h w')
            else:
                img1_ = img1
            img1w = tf_warp(img1_, upflow, data_format)
            _show('6-warp(==0-prv)', img1w[0], data_format)

        if True:
            pred_img1 = model(img_pair)

            # Take the last (full-res) image in case of multi output.
            # This would be the case if e.g. model.output_multiscale==True.
            if multi_output:
                pred_img1 = pred_img1[-1]

            overlay = 0.5 * img0[0] + 0.5 * img2[0]
            _show('0-prv', img0[0], 'channels_last')
            _show('1-nxt', img2[0], 'channels_last')
            _show('2-ground-truth', img1[0], 'channels_last')
            _show('3-pred', 0.5 + pred_img1[0], data_format=data_format)
            _show('4-overlay', overlay, 'channels_last')

        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            break
        continue


if __name__ == '__main__':
    main()
